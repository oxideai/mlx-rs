# `mlx-rs` Contributing Guide

## Orgnaization

### Branches

- `main` is the default branch
- `dev` contains the latest development changes

### Directories

| Directory | Description |
| --- | --- |
| `mlx-sys` | Raw FFI bindings to the `mlx` framework |
| examples | TODO: add examples |

## Developer Guide

### Pre-requisites

1. An Apple silicon Mac
2. Xcode >= 15.0
3. macOS SDK >= 14.0
4. A C++ compiler with C++17 support
5. `cmake` version 3.24 or later and `make`
6. [rust](https://www.rust-lang.org/)

### Building and Testing

1. Clone the repository

    ```sh
    git clone https://github.com/oxideai/mlx-rs.git
    cd mlx-rs && git submodule update --init
    ```

2. Build the project

    ```sh
    cargo build
    ```

3. Run the tests

    ```sh
    cargo test --all
    ```
