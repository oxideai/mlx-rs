# Verification roadmap

Status at `d60f3577` (#391). The 0.32.2 rebind and most verification and API work have landed;
admission still needs stronger evidence before the next mlx-c bump (#387).

## Status

- [x] Reproducible toolchain and FFI fixes (#362, `b7150b8c`), with regressions in
  [ffi_safety.rs](mlx-tests/tests/ffi_safety.rs). MSRV is now 1.88.0 (#366), and CI tests
  1.88.0 and stable; `devenv.nix` selects stable and system Xcode supplies Metal.
- [x] Strict comparator, nine optimizer trajectories, compile-state and nonlinear grad/VJP/JVP
  packs, including Wave 3b (#362). See [state_optimizers.rs](mlx-tests/tests/state_optimizers.rs),
  [state_compile.rs](mlx-tests/tests/state_compile.rs), and
  [state_transforms.rs](mlx-tests/tests/state_transforms.rs).
- [x] Cold-cache subprocess deadline and proof of tracing, stream identity and restoration,
  cross-thread isolation, moved/cloned arrays, and stream churn (#362), in
  [state_compile.rs](mlx-tests/tests/state_compile.rs) and
  [stream_admission.rs](mlx-tests/tests/stream_admission.rs).
- [x] Dependency provenance and runtime rebind to mlx-c `c74db5307cc8ce122f48d97ef951b30578674e7f`
  / MLX `0.32.2` (#362); mlx-rs `0.32.0` / mlx-sys `0.6.0` released in #370.
  [bump_tuple.rs](mlx-tests/tests/bump_tuple.rs) checks the runtime and catchable eval-stage
  singular-inverse error.
- [x] Committed goldens, deterministic subprocess replay, and aggregate admission tooling (#362).
  The current [corpus](conformance/corpus.json) has 260 cases in 11 suites, with Python MLX
  0.32.2 as canonical CPU oracle; NumPy corroboration is recorded per case.
- [x] Idiom migration across operation families, exact/cast observers, keyed `StateProjection`,
  windows and FFT frequencies (#362); GGUF (#363); math and index updates (#364); metallib
  distribution and path APIs (#365); memory controls, contiguous, and optional RMS norm weight
  (#367). See [CHARTER.md](CHARTER.md#idiom-wave-delivery) and
  [the release notes](mlx-rs/CHANGELOG.md#0320).
- [x] mlx-lm oracle/parity gates (#376), local loading, cache, Llama and Qwen3 (#377), generation,
  sampling and prompt-cache reuse (#380), GGUF checkpoints (#382), and Hub/CLI, real-checkpoint
  tests and release evidence (#384, `9051034e`). Executing coverage now lives in
  [parity.rs](mlx-lm/tests/parity.rs) and [the release record](ledger/mlx-lm-release-identity.json).
- [x] Grouped Conv1d/Conv2d weight sizing (#388), native source metadata (#390), Rust 1.99 bindgen
  compatibility (#392), and optional `metal-jit` (#391) have merged; see
  [convolution.rs](mlx-rs/src/nn/convolution.rs) and [build.rs](mlx-sys/build.rs).

<a id="tranche-2-leak-and-use-after-free-gate-done"></a>

## Tranche 2 — leak and use-after-free gate (done)

    cargo run -p xtask -- verify-ffi
    cargo run -p xtask -- verify-ffi --guard-malloc

The gate implementation and deliberate-leak calibration landed in #362. The heading is retained
for admission evidence links; continuous use-after-free enforcement is still open.

For local admission, record the exact SHA and environment with both reports. Delete the body of
`StringMapIterator::drop` in `mlx-rs/src/utils/io.rs`, confirm the iterator leak is detected, then
restore it and require zero regressions. Requalify after parser or baseline changes.

`leaks --atExit` reports a constant Objective-C initialization leak (32 bytes), so compare against
the qualified baseline rather than trusting its exit code. Discover binaries through Cargo JSON;
exclude proc-macro binaries by target kind. Guard Malloc supplies the separate use-after-free and
double-free lane. CI calibrates leaks, but `verify(..., false, ...)` in
[verify_ffi.rs](xtask/src/verify_ffi.rs) disables Guard Malloc during calibration.

## Conformance runner

    cargo test -p mlx-tests --test conformance -- --test-threads=1

The qualified strict comparator checks dtype, shape, output count/order, separate tolerances,
non-finite values, and layout-sensitive observations. Regeneration uses Python 3.12.14,
`mlx==0.32.2`, `mlx-metal==0.32.2`, and `numpy==2.2.6`, pinned in
[requirements.lock](conformance/requirements.lock); it requires identical hashes from two runs.

[replay_target.py](conformance/target/replay_target.py) implements deterministic subprocess replay
with an exact-version handshake. Its committed reports preserve the historical 0.30.6-to-0.32.2
comparison. Scheduled randomized exploration, deadlines, shrinking, and case promotion remain open.

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

The delivered bump moved from mlx-c `a1290d221f92bd020af805b7d14207eee4ec973b` / MLX `0.30.6`
to `c74db5307cc8ce122f48d97ef951b30578674e7f` / MLX `0.32.2` in #362. The eight admission
requirements remain:

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
a semantic regression. New public cohorts land separately with mapping, independent expectations,
ownership and applicable thread/state evidence, and documentation under the
[charter](CHARTER.md#charter-rules).

## Remaining work

Step 1, reconciling this roadmap and the signed charter, is complete here. Step 2 is required
before #387; step 3 follows in the order below.

1. Step 2: enforce compiled, feature-qualified API coverage, including generated builders and cfg
   availability, and run `verify-ledger` in CI; the current source inventory omits both ([api_baseline.rs](xtask/src/api_baseline.rs), [validate.yml](.github/workflows/validate.yml)). Size: medium.
2. Step 2: fingerprint linked-library exports and compare them with the planned ABI; generation
   currently reads only header-derived bindgen output ([fingerprint.rs](xtask/src/fingerprint.rs)). Size: medium.
3. Step 2: require build/test reports for CPU debug/release and default Metal/Accelerate, plus
   dependency canaries and `metal-jit` configuration checks; admission currently counts two declared
   builds, and CI tests default debug features on MSRV/stable ([verify_bump.rs](xtask/src/verify_bump.rs), [validate.yml](.github/workflows/validate.yml), [supported-feature-matrix.json](ledger/supported-feature-matrix.json)). The CPU-only
   build cannot be produced yet: the workspace `mlx-sys` dependency keeps its default features, so
   `--no-default-features` on `mlx-rs` still builds Metal ([Cargo.toml](Cargo.toml)). Size: medium.
4. Step 2: enforce and qualify Guard Malloc continuously so CI supplies the exact-SHA report already
   required by bump admission; calibration currently disables it ([verify_ffi.rs](xtask/src/verify_ffi.rs), [verify_bump.rs](xtask/src/verify_bump.rs)). Size: medium.
5. Step 3: clear test-code lint failures and gate `--all-targets`, triage the 47 ignored doctests into
   compiling examples or text, and deprecate or redesign the dtype-blind, coupled-tolerance
   `assert_array_eq!` ([validate.yml](.github/workflows/validate.yml), [linalg.rs](mlx-rs/src/linalg.rs), [activation.rs](mlx-rs/src/nn/activation.rs), [assert.rs](mlx-rs/src/macros/assert.rs)). Size: small each.
6. Step 3: add independent asymmetric FFT and `full` goldens; FFT currently checks DC and round trips,
   while `full` expectations are constant ([fftn.rs](mlx-rs/src/fft/fftn.rs), [factory.rs](mlx-rs/src/ops/factory.rs), [corpus.json](conformance/corpus.json)). Size: medium.
7. Step 3: finish the observer and canonical-home contracts; `try_as_slice` still evaluates
   implicitly, and method/free-function duplicates remain ([array/mod.rs](mlx-rs/src/array/mod.rs), [shapes.rs](mlx-rs/src/ops/shapes.rs), [charter rules](CHARTER.md#charter-rules)). Size: large.
8. Step 3: add scheduled seeded differential exploration with subprocess deadlines, failure
   shrinking, and reviewed promotion into the corpus ([replay_target.py](conformance/target/replay_target.py), [validate.yml](.github/workflows/validate.yml)). Size: large.

## Deferred and cut work

- Shift APIs remain blocked on defined invalid-count semantics at a future pin; recheck during the
  next bump, without implicitly evaluating lazy counts. Investigation size: small; evidence is
  [the C wrappers](mlx-sys/src/mlx-c/mlx/c/ops.cpp).
- The [C-delta ledger](ledger/target-delta-classification.json) has 18 deferred entries: distributed,
  diagnostics, fast cross-entropy/gather-QQMM, and the deliberately unexposed unsafe stream.
  Distributed needs a use case and multi-process ownership evidence; diagnostics need a consumer;
  fast model ops need model, dtype, gradient, and quantization evidence; the unsafe stream needs a
  provable safe ownership contract. These are not unconditional debt.
- The [mlx-lm ledger](ledger/mlx-lm-parity.json) has 198 deferrals with explicit triggers, including
  batching, adapters, and quantized/persisted caches. Expand only an approved cohort with its
  ownership and parity evidence; the delivered model wave creates no universal parity obligation.
- Full Python parity and historical C-to-Rust mapping remain cut; a designated module cohort can
  justify a scoped ledger. Internal FFT normalization and intentionally omitted convenience APIs
  retain their [charter dispositions](CHARTER.md#disposition-of-the-81-initially-deferred-ledger-entries).
- Executing mathematical pseudocode as Rust is cut. Full ASan instrumentation needs a spike showing
  findings distinct from leaks/Guard Malloc; parser or run-count work needs a demonstrated diagnostic
  or runtime problem and must requalify the FFI gate.
