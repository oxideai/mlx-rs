use autocxx::prelude::*;

include_cpp! {
    #include "mlx/mlx.h"
    #include "extras.h"
    // TODO: what safety option should be used here?
    safety!(unsafe)
    generate!("hello")
}

#[cfg(test)]
mod tests {
    #[test]
    fn extras_hello_works() {
        let hello = super::ffi::hello();
        println!("ffi::hello: {}", hello);
    }
}