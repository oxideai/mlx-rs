use autocxx::prelude::*;

include_cpp! {
    #include "mlx/mlx.h"
    safety!(unsafe)
    // mlx/mlx/random.h
    generate_ns!("mlx::core::random")
}

fn main() {
    let key = ffi::mlx::core::random::key(1).within_box();
}