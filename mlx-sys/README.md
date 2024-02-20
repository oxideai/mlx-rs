# mlx-sys

DO NOT USE. This is an experimental crate for testing right now, and it is not intended
to be used directly. A separate safe wrapper crate will be created in the future.

---

- [Documentation](https://minghuaw.github.io/mlx-rs/doc/mlx_sys/)

## Naming

- Numeric types that are not in the rust standard library will use whatever the MLX library uses, eg. `float16_t`
- Trivial C++ types, if not following the rust naming convention, will be renamed to follow the rust naming conventions. This includes enums and structs, eg. `DeviceType::Cpu`.
- Opaque C++ types will use whatever the MLX library uses, eg. `array`.

## Exception and Result

The overall strategy for catching exceptions and turning them into Rust errors is

1. Ignore memory allocation exceptions. This is consistent with the behavior of `Vec` in rust.
2. `load_library` or `get_kernel`, this would include all ops and usually indicates a problem with the library, so we should probably just let it panic
