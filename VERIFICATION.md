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

CI runs this gate on every PR and push to main, uploading reports and logs even on failure. Reports
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
with an exact-version handshake. Its committed reports compare MLX 0.30.6 with 0.32.2.

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

The source-based ledger, header/bindgen fingerprints, executed configuration reports, and aggregate
`verify-bump` command exist. Configuration admission requires passing reports for every required
ledger entry at the admitted SHA. These reports do not establish compiled API coverage or
linked-library exports; leak calibration alone does not establish the required Guard Malloc lane.
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
workflow. Its focused evidence verdict supplements the aggregate admission contract, including the
separate qualified FFI gate; it is not a claim that all eight requirements passed.

Freeze the old and target commits and nested MLX version before generating the isolated header,
bindgen, ABI-type, and linked-export comparison. Classify every delta, update provenance together,
and verify the linked runtime.

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

The workflow runs stable configurations serially, then native canaries and MSRV with the same
checkpoint manifest. It validates all five configuration reports and runs the focused check:

    cargo run -p xtask -- verify-bump --bump-evidence-only \
      --config-report "$BUMP_ROOT/reports/macos-arm64-default.json" \
      --config-report "$BUMP_ROOT/reports/macos-arm64-metal-jit.json" \
      --native-report "$BUMP_ROOT/reports/native/native-cpu-debug.json" \
      --native-report "$BUMP_ROOT/reports/native/native-cpu-release.json" \
      --native-report "$BUMP_ROOT/reports/native/native-metal.json"

Aggregate `verify-bump --pre` / `--admit` also require those native reports via repeated
`--native-report` flags, alongside all required `--config-report` and qualified FFI evidence.
Missing, duplicate, stale, dirty, failed or empty native evidence fails; the resolved MLX commit
must agree with the default and JIT Rust builds. Ordinary PR validation has no new requirement.

Run qualified leak and Guard Malloc lanes, old semantic baselines before changing oracle authority,
exact-target replay, affected high-risk packs and public-surface gates. Aggregate exact-SHA,
environment-bound reports and waivers into one verdict; never regenerate away a semantic regression.
Keep checkpoint caches and reports outside the checkout. Delete isolated native, Cargo and metallib
build directories after collecting evidence. Reports record measured times; the workflow has a
180-minute limit to accommodate cold builds and three CPU/Metal checkpoint passes.

New public APIs land vertically, as their own change under the [charter](CHARTER.md#charter-rules):
parity mapping, independently sourced cases with a golden or differential result, ownership
classification, applicable thread/state and property or gradient evidence, and public documentation.
