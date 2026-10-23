# Verification

## FFI leak and use-after-free gate

    cargo run -p xtask -- verify-ffi
    cargo run -p xtask -- verify-ffi --guard-malloc

For admission, run the qualified combined gate from a clean checkout and put its artifacts outside
that checkout:

    mkdir -p "$RUNNER_TEMP/ffi-gate"
    cargo run -p xtask -- verify-ffi --calibrate --guard-malloc \
      --budget-seconds 3600 --leaks-timeout-seconds 600 \
      --guard-malloc-timeout-seconds 1200 \
      --report "$RUNNER_TEMP/ffi-gate/verify-ffi.json" \
      > "$RUNNER_TEMP/ffi-gate/calibration.json"

Locally, use a scratch directory in place of `$RUNNER_TEMP/ffi-gate`. Pass `verify-ffi.json` to
`verify-bump --ffi-report`; local evidence also requires `--allow-local-ffi`. The calibration JSON
on stdout contains the detailed controls. Requalify after parser or baseline changes.

`leaks --atExit` reports a constant Objective-C initialization leak (32 bytes), so compare against
the qualified baseline rather than trusting its exit code. Discover binaries through Cargo JSON;
exclude proc-macro binaries by target kind. Guard Malloc supplies the separate use-after-free and
double-free lane. Calibration must detect a deliberate iterator leak, use-after-free, and double-free,
and pass clean controls before its combined inventory report can qualify admission. Each Guard Malloc
process must show that libgmalloc loaded; unavailable instrumentation and timeouts fail the gate.

CI runs the calibrated leak gate on every PR, proving that leaks can attach and detect the deliberate
leak. Pushes to main also run Guard Malloc. For PRs changing the `mlx-sys/src/mlx-c` gitlink and
`workflow_dispatch`, `bump-admission.yml` runs the full qualified leaks + Guard Malloc gate in a
separate job alongside bump evidence. Each gate logs its mode and reason and uploads
`verify-ffi-calibration` reports and logs even on failure. Use the bump-admission artifact for
bump PR admission; the leak-only validation report does not satisfy `verify-bump`, which still
requires qualified Guard Malloc evidence for the admitted SHA. Reports
cover `source_commit`, the checked-out commit (the merge commit on PRs). They also record `github_sha`
and `github_pr_head_sha` when available; the PR head is context, not the admitted commit. The clean
gate has a 60-minute budget, each leak subprocess a 10-minute timeout, and each Guard Malloc
subprocess a 20-minute timeout. The calibration step has a 75-minute timeout including builds.

## Conformance runner

    cargo test -p mlx-tests --test conformance -- --test-threads=1

The qualified strict comparator checks dtype, shape, output count/order, separate tolerances,
non-finite values, and layout-sensitive observations. Regeneration uses Python 3.12.14,
`mlx==0.32.2`, `mlx-metal==0.32.2`, and `numpy==2.2.6`, pinned in
[requirements.lock](conformance/requirements.lock); it requires identical hashes from two runs.

[replay_target.py](conformance/target/replay_target.py) implements deterministic subprocess replay
with an exact-version handshake. It discovers all corpus suites and also runs state, transforms,
and the historical probes. Its committed reports compare MLX 0.30.6 with 0.32.2;
[scratch replay](conformance/README.md#exact-target-replay) compares the current corpus without
replacing those reports.

### Anti-self-oracle rules

1. The corpus generator must never link or execute `mlx-rs`.
2. Every expected value carries provenance: a NumPy expression, a pinned Python MLX version, an
   upstream literal, or a named mathematical invariant.
3. An implementation change may not modify goldens, tolerances, or comparator logic in the same
   commit. Oracle changes are reviewed on their own.
4. Tolerances come from a central registry. Widening one is an oracle change.

Python stays out of the Rust process because both runtimes carry global caches. CPU is canonical;
Metal comparisons are separate. Use a scratch `CARGO_TARGET_DIR` for different feature sets.

## Executed configuration reports

The required entries in `ledger/supported-feature-matrix.json` identify the default stable/MSRV,
CPU-only stable debug/release and Metal JIT stable debug lanes, including package selection,
Rust profile, toolchain, resolved backend features and native CMake settings. Run each configuration with its own
`CARGO_TARGET_DIR` and put reports outside the checkout:

    cargo run -p xtask -- config-report macos-arm64-default \
      --report "$RUNNER_TEMP/config-reports/macos-arm64-default.json"
    cargo run -p xtask -- config-report macos-arm64-cpu-only \
      --report "$RUNNER_TEMP/config-reports/macos-arm64-cpu-only.json"
    cargo run -p xtask -- config-report macos-arm64-cpu-only-release \
      --report "$RUNNER_TEMP/config-reports/macos-arm64-cpu-only-release.json"
    cargo run -p xtask -- config-report macos-arm64-metal-jit \
      --report "$RUNNER_TEMP/config-reports/macos-arm64-metal-jit.json"

Use the selected toolchain for the entire invocation; `macos-arm64-default-msrv` requires Rust
1.88.0. Give each configuration a separate `MLX_RS_METAL_PATH` as well; JIT and non-JIT
metallibs must not share a directory. Locally, substitute a scratch directory for `$RUNNER_TEMP`.
The wrapper executes the ledger's Cargo test command once, including doctests, with serial tests
and captured test output.
It preserves Cargo's exit status and writes a JSON report plus a sibling `.log` file. A zero exit
status does not qualify evidence: inspect `verdict`, or use the admission validator.

Reports bind the clean checkout SHA, mlx-c commit, nested MLX version, Cargo.lock SHA-256,
`rustc --version --verbose`, macOS/SDK versions, resolved features, Rust profile and compiled
optimization/debug-assertion settings, effective CMake cache settings (including build type and `MLX_METAL_JIT`), exact argument vector,
build-plus-test wall time, and each binary/doctest's counts and outcome.
`github_sha` and `github_pr_head_sha` are context; only `source_commit` is admitted. Missing runs,
incomplete summaries, filtered tests, `NOT RUN`, and unexpected zero-test runs fail qualification.
The ledger's `allow_empty` lists intentionally empty and compile-only harnesses, the ignored leak
calibration probe, and documentation targets without runnable tests. Ignored individual tests remain
visible in counts. Nested harness summaries are recorded separately from the enclosing binary
so a filtered child regression probe does not inflate its parent's counts.

Pass one `--config-report <file>` per required configuration to `verify-bump`. It rejects missing,
duplicate, stale, failed or skipped evidence and configuration mismatches. To run only this admission
check against the current checkout:

    cargo run -p xtask -- config-report --verify "$RUNNER_TEMP/config-reports/"*.json

CI uploads these reports and logs even when tests fail. Ordinary PR runs do not download
checkpoints and their workspace reports retain `NOT RUN` when the manifest is absent. Bump runs
supply the pinned fixtures and independent Python expectations described below. Add a ledger entry
and a CI invocation for each new configuration; admission discovers required entries without a
fixed configuration count.

## Compiled public API inventory

Install the extraction toolchain without changing the active toolchain:

    rustup toolchain install "$(cat xtask/api-nightly.txt)" --profile minimal
    cargo run -p xtask -- api-baseline --check ledger/rust-api-baseline.json
    cargo run -p xtask -- verify-ledger

`xtask/api-nightly.txt` pins nightly for rustdoc JSON extraction only. `xtask` builds on stable;
`public-api = 0.50.0` normalizes JSON schema 55. Extraction requires an `aarch64-apple-darwin`
host and uses the pinned compiler and rustdoc explicitly. Use `MLX_RS_API_TARGET_DIR` for scratch
builds and per-configuration JSON files, and `MLX_RS_METAL_PATH` for scratch metallibs. The default
build directory is `target/api-inventory`. Nightly targets are separate from stable targets;
CPU and Metal builds are separate, while adding `safetensors` reuses its backend's dependencies.

The scope is `mlx-rs`. The four API feature sets are `default`, `cpu-only`
(`--no-default-features --features accelerate`), `safetensors` (default plus `safetensors`),
and `cpu-only-safetensors`. Extraction also compares `metal-jit` with and without `safetensors`
against the corresponding non-JIT inventory and fails if their public surfaces differ, requiring
a distinct feature set before regeneration. JIT metallibs use a separate `metal-jit` subdirectory.

The committed baseline merges identical entries and lists their feature-set availability.
It sorts entries by path and declaration and writes one JSON entry per line. Each entry records
the public path, kind, normalized signature, generated and hidden flags, and the enclosing impl
header, deprecation metadata, and procedural-macro helper attributes where applicable. Simple
local type aliases without generic parameters or target arguments record their compiled target; member entries retain canonical paths, and ledger
lookup resolves those alias prefixes with the intersection of alias and member availability.
Specialized aliases remain unexpanded because their method bounds require substitution. Impl
declarations and generic receiver members use the impl header as
their identity because they have no single crate-rooted Rust path. Public re-exports are expanded
by rustdoc and `public-api`. The adapter fills external procedural-macro globs from separately
compiled dependency JSON because rustdoc leaves those targets unresolved. Other unresolved
re-exports fail extraction rather than leaving wildcard entries. Source parsing only classifies
provenance against compiled spans.
It does not discover or admit inventory entries. A generated flag means the declaration was
emitted by a macro, derive, or compiler, rather than a matching declaration at its source span.

Blanket implementations and their members are omitted. Auto-trait implementations, including
thread traits, and implementations derived for mlx-rs types remain tracked.

Public `#[doc(hidden)]` items remain included. Their flags come from their attributes and a
comparison of public item identities with a second rustdoc run without hidden items, which also
covers hidden re-exports without confusing a hidden field with its visible containing type.
Every extraction qualifies the adapter with compiled builder, derive, re-export, hidden-public,
and cfg fixtures. Five separately compiled omission controls (including an external derive
re-export), two compiled Send/Sync change controls, and 24 incorrect-field controls must fail.
Stable consumer probes compile the fixture in both configurations. Run just these controls with
`api-baseline --qualify`.

`verify-ledger` regenerates the compiled inventory and rejects baseline drift before checking
wrapped paths, behavioral renames, admission paths, symbol mappings, and `api:path:` evidence.
An unqualified reference claims every API feature set. An object's `api_feature_sets` list
narrows that claim for its references and descendants; unknown, empty, or duplicate sets fail.
Metal-only references explicitly claim `default` and `safetensors`. A `wrapped` entry with no
supported configuration can cite `unsupported_cfg` as
`api:cfg:<source-file>#<function>#not(target_os = "macos")`, repeat that citation in `evidence`,
and supply a one-line `rationale`. This classification currently accepts only the canonical
`mlx_rs::ops::qqmm` path and its `mlx-rs/src/ops/quantization.rs` citation. It requires a macOS
inventory target, checks that the named public function carries the exact cfg, and rejects a
path present in any supported feature set. It does not admit that wrapper to the supported surface.

The separate `mlx-lm` parity check still uses `xtask/src/source_api.rs`; it is not compiled API
evidence for `mlx-rs`.

Regenerate with `api-baseline --out ledger/rust-api-baseline.json` and review the diff. CI runs
extraction, its qualification controls, `--check`, and ledger verification in the existing
`checks` job. A baseline match establishes compiled surface coverage, not semantic correctness
or linked-library exports.

## Bump admission contract

Every mlx-c bump must meet these eight admission requirements:

1. An immutable target tuple and checked-in playbook.
2. Oracle boundary, digest integrity, and staged-case enforcement.
3. Error-registration safety and concurrent regressions.
4. Exact-SHA qualified leak and Guard Malloc verdicts, in CI or an explicitly lower-trust local
   admission record.
5. Strict tensor comparison and meaningful nonzero optimizer/compile transitions.
6. Classified C/ABI delta, compiled feature-qualified Rust API coverage, linked exports, and
   evidence that supported configurations built and ran.
7. Qualified optimizer, compile-state, and transform packs.
8. Deterministic replay against the exact target Python MLX version.

The compiled feature-qualified API inventory, ledger, header/bindgen and linked-archive fingerprints, executed configuration reports, and aggregate
`verify-bump` command exist. Configuration admission requires passing reports for every required
ledger entry at the admitted SHA. Native export reports bind archive bytes and retained-symbol link/version canaries to the planned C functions.
Leak calibration alone does not establish the required Guard Malloc lane.
The presence of the tooling is not evidence that the whole admission playbook passed.

Ledger evidence stays typed: `conf:` for semantic cases, `ffi:` for ABI/ownership/lifetime evidence,
`state:` for trajectories and atomicity, `thread:` for concurrency, and `api:` for public paths and
build contracts. Semantic cases alone cannot prove ownership or threading safety.

Each waiver names one check or entry, its missing capability, risk, compensating evidence, owner,
reviewer approval, and expiry no later than the next bump. Missing, blanket, unreviewed, or expired
waivers fail admission; every active waiver appears in the aggregate verdict.

### Repeatable `mlx-c` bump playbook

`.github/workflows/bump-admission.yml` runs on PRs changing the `mlx-sys/src/mlx-c` gitlink
and on `workflow_dispatch`, on `blaze/macos-latest`. A PR run admits the checked-out merge SHA;
a dispatch admits the selected ref's SHA. Native canaries and checkpoint downloads run only in this
workflow. It runs the qualified FFI gate, collects exact-SHA evidence, and ends verification with
the aggregate `verify-bump --admit` verdict for all eight admission requirements.

Freeze the old and target commits and nested MLX version before generating the isolated header,
bindgen, ABI-type, and linked-export comparison. Classify every delta, update provenance together,
and verify the linked runtime.

The active comparison is mlx-c `c74db5307cc8ce122f48d97ef951b30578674e7f` to
`a341b4925024b88b2c593468f16e12f5e17315da`; both pin MLX `v0.32.2`.
`verify-bump --pre` checks the old checkout; after moving the submodule, use `--admit`
with the old-tuple boundary base and `--allow-local-ffi` for local evidence.
The prior delta classification is retained in
`ledger/mlx-c-c74db5307cc8ce122f48d97ef951b30578674e7f-classification.json`.

`python3 -B xtask/provision_checkpoints.py --cache "$BUMP_ROOT/checkpoints" --output
"$BUMP_ROOT/reports/checkpoints"` downloads 6,199,630,244 bytes across six checkpoints and two
tokenizer directories. `conformance/mlx-lm/bump-oracle.json` pins every repository revision and
file SHA-256. The cache key hashes that file, and every restored file is rehashed before use.
Downloads use immutable revision URLs and temporary files; mismatches fail before installation.
The provisioner reuses `real_checkpoints.py`'s manifest validation and relocates only local paths in
the frozen Python report. It never generates expected tokens from Rust.

The frozen report comes from the independent MLX 0.32.2 / mlx-lm 0.31.3 host run recorded in
`mlx-lm/SKELETON.md`; the original report's SHA-256 is
`a90e92d9a676b50040ddd097395ae4c53695687422f2df0a8080cc3380cd14fa`.
Only `contract.entries.*.local_directory` was normalized for portability. Token IDs, logits,
configuration, provenance and existing Metal tie rules are unchanged. Updating these expectations
is an oracle change and must remain separate from implementation changes.

Set `MLX_LM_REAL_CHECKPOINTS="$BUMP_ROOT/reports/checkpoints/real-checkpoints.json"` and
`MLX_RS_BUMP_RUN=1`, then run `config-report` for the five required configurations with separate
`CARGO_TARGET_DIR` and `MLX_RS_METAL_PATH` directories. The bump flag permits the Rust checkpoint
harness in CI; the benchmark and Python regeneration harness remain local-only. The Rust test still
hashes all files and the independent report and executes all six cases on CPU and Metal. No manifest
still means `NOT RUN`. Default and JIT reports also bind the frozen oracle digest and all twelve
case/device executions. A changed expectation, missing case or failed comparison blocks admission.

Run `python3 -B xtask/native_canaries.py --build-root "$BUMP_ROOT/native" --reports
"$BUMP_ROOT/reports/native"` with an isolated build directory. Reuse is allowed only for a clean nested source at the same tag. This separate CMake project builds the
checked-out mlx-c and its pinned nested MLX tag, records the resolved MLX commit, and adds upstream's
own test target despite mlx-c disabling it. It does not modify upstream sources or Cargo's builds.
The three native configurations are CPU/Accelerate debug, CPU/Accelerate release, and
Metal/Accelerate release with JIT off. Upstream enables Accelerate when its framework is found, even
if the cache says `MLX_BUILD_ACCELERATE=OFF`; native evidence checks the compiled definition.
`xtask/native-canaries/plan.json` names the canaries: all eleven C examples built, with seven CPU-capable examples run in every configuration
and the gradient, closure, export and Metal-kernel examples run on Metal; eleven C++ cases covering shared arrays, lazy evaluation, device
and stream placement, gradients/VJP and compilation across streams; and Metal matmul, reduction and
binary operations. These exercise the C boundary and upstream execution paths used by the Rust
library without running the entire upstream suite. Each named C++ case must execute exactly once
with nonzero passing assertions. Reports retain commands, compiler/SDK, effective CMake settings,
case counts, build/run timings and source identities; XML and logs are uploaded on failure too.

Each native build also runs `linked-exports`. CMake resolves the archive targets named in
`mlx-sys/native-libraries.txt`, the same list consumed by `mlx-sys/build.rs`. JACCL is included only
when CMake produces its archive; native builds select it when the `jaccl` target exists. The checker
fingerprints only the archives built from the pinned sources: libmlx, libmlxc, libgguflib, and
libjaccl when built. The compiler runtime comes from the Xcode toolchain resolved by mlx-sys's
`find_clang_rt_path`, independently of the native build manifest. That producer lookup, rather than
an archive-name prefix or location inside the build tree, identifies the toolchain archive.
Reports record its path and SHA-256 in `toolchain_archives` without enumerating or comparing its
symbols; the canary still links it. These are the native builds' artifacts, not Cargo's separately
compiled bytes. Reports under `native/exports/` record exact paths,
SHA-256 hashes, CMake configuration, the planned functions from the existing header/bindgen parser,
and defined external symbols in each archive's arm64 slice. Mach-O's leading underscore is removed
once. C++ Itanium-mangled symbols are counted separately; every other defined external symbol,
including weak symbols and data, is conservatively treated as a C export. This inventories archive
members, including members a particular consumer may not pull into its executable.

`ledger/native-export-exceptions.json` records reviewed exact-name exceptions for gguflib, compiler
helpers, and metal-cpp data globals. Each entry names its defining archive, a one-line reason, and
whether it applies only when `MLX_BUILD_METAL=ON`. Missing planned symbols, unlisted exports,
exceptions used outside their archive or configuration, and unused exceptions in an applicable
configuration fail admission. An exception for an optional archive applies only when that archive
is built. Reports retain unexpected symbols grouped by archive even on failure;
adding exceptions requires a reviewed policy change, not automatic baseline regeneration.
Header/type fingerprints remain necessary because a matching symbol name does not prove its type.

The generated C++ canary includes the same headers, retains a volatile reference to every planned
function, checks the object's undefined symbols, and links those exact archives. Its runtime smoke
check calls both `mlx_version` through mlx-c and `mlx::core::version` and requires the pinned MLX
version from both. mlx-c has no separate runtime version string: its C version API returns MLX's
version. The mlx-c identity is its source commit. Link and smoke results are retained separately
from the export verdict, and unexpected exports do not skip the native test cases.

To inspect an existing native build, or Cargo's actual build-script link resolution:

    cargo run -p xtask -- linked-exports --build "$BUMP_ROOT/native/native-cpu-debug" \
      --report "$BUMP_ROOT/reports/exports.json"
    cargo check -p mlx-sys --message-format=json > "$BUMP_ROOT/reports/cargo-artifacts.json"
    cargo run -p xtask -- linked-exports --cargo-output "$BUMP_ROOT/reports/cargo-artifacts.json" \
      --report "$BUMP_ROOT/reports/cargo-exports.json"

`linked-exports --verify <report>` rehashes and rescans the archives and checks the current planned
symbols, CMake configuration/source provenance, and canary binary hash. Archives must remain inside
the selected build directory; compiler-runtime archives are resolved separately. Run admission before deleting build directories; retained reports
are a review record, not evidence that can be revalidated after their archives are gone.

After the FFI job finishes, the evidence job runs stable configurations serially, then native
canaries and MSRV with the same checkpoint manifest. It restores stable Rust, installs the pinned
API extraction nightly, and records conformance, optimizer, compile-state and transform suite
results and logs. The final aggregate uses the Rust compiler version recorded in both committed
ABI fingerprints; the job installs its rustfmt component and verifies the full compiler version
before admission. It consumes those reports and the FFI job's report, validates
the committed replay, rebuilds both ABI fingerprints, and checks the compiled API and ledger.
Native archives remain available until this verdict completes. All reports and logs upload even
on failure. PRs use the PR base as the oracle boundary; dispatches use the parent of the latest
submodule change, which must record the old tuple.

To check just configuration, native and checkpoint evidence locally:

    cargo run -p xtask -- verify-bump --bump-evidence-only \
      --config-report "$BUMP_ROOT/reports/macos-arm64-default.json" \
      --config-report "$BUMP_ROOT/reports/macos-arm64-metal-jit.json" \
      --native-report "$BUMP_ROOT/reports/native/native-cpu-debug.json" \
      --native-report "$BUMP_ROOT/reports/native/native-cpu-release.json" \
      --native-report "$BUMP_ROOT/reports/native/native-metal.json"

Aggregate `verify-bump --pre` / `--admit` also require those native reports via repeated
`--native-report` flags, alongside all required `--config-report` and qualified FFI evidence.
Missing, duplicate, stale, dirty, failed or empty native evidence fails; the resolved MLX commit
must agree with the default and JIT Rust builds. The `checks` job also inspects the current Cargo
archives discovered from clippy's JSON build-script output, reusing that job's native build. It
uploads the export report and Cargo artifact inventory on failure too. The measured local addition
was 20 seconds including rebuilding xtask, below the three-minute budget; no native rebuild or
new routine job is added. Any export outside the reviewed exception scopes fails this
check and bump admission.

Run qualified leak and Guard Malloc lanes, old semantic baselines before changing oracle authority,
exact-target replay, affected high-risk packs and public-surface gates. Aggregate exact-SHA,
environment-bound reports and waivers into one verdict; never regenerate away a semantic regression.
Keep checkpoint caches and reports outside the checkout. Delete isolated native, Cargo and metallib
build directories after collecting evidence. Reports record measured times; the workflow has a
240-minute evidence-job limit to accommodate cold builds, three CPU/Metal checkpoint passes,
semantic/state suites, and aggregate API and fingerprint verification.

New public APIs land vertically, as their own change under the [charter](CHARTER.md#charter-rules):
parity mapping, independently sourced cases with a golden or differential result, ownership
classification, applicable thread/state and property or gradient evidence, and public documentation.
