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

CI uploads these reports and logs even when tests fail. The current default workspace includes
`mlx-lm`'s local-only `real_checkpoints` test: without its manifest it prints `NOT RUN`, so default
and Metal JIT reports cannot qualify admission even when Cargo succeeds. This report exposes that
execution gap; it does not supply checkpoint fixtures or add a CI checkpoint lane. Native dependency
canaries remain a separate addition. Add a ledger entry and a CI invocation for each new configuration; admission discovers required entries without a fixed configuration count.

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

Freeze the old and target commits and nested MLX version before generating the isolated header,
bindgen, ABI-type, and linked-export comparison. Classify every delta, update provenance together,
and verify the linked runtime. Build CPU debug/release and default Metal/Accelerate canaries,
including mlx-c examples and an upstream MLX C++ test subset.

Run qualified leak and Guard Malloc lanes, old semantic baselines before changing oracle authority,
exact-target replay, affected high-risk packs, public-surface gates, and the supported workspace
matrix. Include MSRV/stable, threading cases, genuine doctests, and offline model smoke tests.
Aggregate exact-SHA, environment-bound reports and waivers into one verdict; never regenerate away
a semantic regression.

New public APIs land vertically, as their own change under the [charter](CHARTER.md#charter-rules):
parity mapping, independently sourced cases with a golden or differential result, ownership
classification, applicable thread/state and property or gradient evidence, and public documentation.
