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

The source-based ledger, header/bindgen fingerprints, declared feature matrix, and aggregate
`verify-bump` command exist. They do not establish compiled API coverage, linked-library exports,
or matrix execution; leak calibration alone does not establish the required Guard Malloc lane.
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
