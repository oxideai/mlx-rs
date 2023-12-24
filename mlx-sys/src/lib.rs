use autocxx::prelude::*;

include_cpp! {
    #include "mlx/mlx.h"
    #include "extras.h"
    // TODO: what safety option should be used here?
    safety!(unsafe)
    generate!("extra::hello")
    generate!("mlx::core::array")
    generate!("mlx::core::Device")
}

#[cfg(test)]
mod tests {
    #[test]
    fn extras_hello_works() {
        let hello = super::ffi::extra::hello();
        println!("ffi::hello: {}", hello);
    }
}