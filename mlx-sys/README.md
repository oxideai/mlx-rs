# mlx-sys

DO NOT USE. This is an experimental crate for testing right now, and it is not intended
to be used directly. A separate safe wrapper crate will be created in the future.

## Exception and Result

The overall strategy for catching exceptions and turning them into Rust errors is

1. Ignore memory allocation exceptions. This is consistent with the behavior of `Vec` in rust.
2. `load_library` or `get_kernel`, this would include all ops and usually indicates a problem with the library, so we should probably just let it panic
