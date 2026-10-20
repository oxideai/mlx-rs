## Building and testing

Use the repo's devenv environment, activated through direnv with `direnv allow`.
Run tools directly once inside it. Build with `cargo build --workspace`.
Run tests with `cargo test --workspace -- --test-threads=1`: MLX's global state
is not safe for parallel tests.

The devenv script `validate` runs fmt, clippy, and the full workspace test suite.
`validate-fast` runs the same checks without doctests.
CI uses the latest stable Rust, so new lints can appear there before they appear locally.

## Before a PR

Run these checks as well as the tests for your change:

```sh
cargo fmt -- --check
cargo clippy -- -D warnings
cargo run -p xtask -- verify-oracle-boundary --base main
cargo run -p xtask -- verify-ledger
cargo run -p xtask -- api-baseline --check ledger/rust-api-baseline.json
```

When touching FFI ownership, also run `cargo run -p xtask -- verify-ffi`.

## Rules for every change

- An implementation change must not share a commit with oracle changes to goldens,
  tolerances, or comparator logic.
- Add no new `_device` twins or `generate_macro` operation macros.
- Use `with_stream` and `with_device` for stream selection.
- Commit subjects follow the existing `type(scope): subject` history.

## Read before changing

- Read [CHARTER.md](CHARTER.md) before adding or changing public API.
- Read [VERIFICATION.md](VERIFICATION.md) before touching `conformance/`, `ledger/`,
  the FFI gate, or an mlx-c bump.
