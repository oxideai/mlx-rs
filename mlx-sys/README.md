# mlx-sys

DO NOT USE. This is an experimental crate for testing right now, and it is not intended
to be used directly. A separate safe wrapper crate will be created in the future.

## Exception and Result

The overall strategy for catching exceptions and turning them into Rust errors is

1. Ignore memory allocation exceptions. This is consistent with the behavior of `Vec` in rust.
2. Ignore exceptions thrown in operator overloads. Because these are not exposed but their underlying equivalent are exposed. However, many of the checks are performed in the overloaded operators, so this may need to be revisited.
3. `load_library` or `get_kernel`, this would include all ops and usually indicates a problem with the library, so we should probably just let it panic
