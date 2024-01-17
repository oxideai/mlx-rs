//! The function pointer used in functions like `grad()` are used to
//! eventually compute the output `array` which is then used to compute the gradient.
//! So theoretically we could instead pass a rust function that is callable from C++.

use crate::function::UnaryFn;

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/array.h");
        include!("mlx-cxx/transforms.hpp");

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        #[namespace = "mlx_cxx"]
        type UnaryCxxFn;

        #[namespace = "mlx_cxx"]
        type MultiaryCxxFn;

        // TODO: This clearly changes internal states of the arrays. We should review if it should
        // be put behind a mut reference.
        #[namespace = "mlx_cxx"]
        fn simplify(outputs: &[UniquePtr<array>]);

        // TODO: This clearly changes internal states of the arrays. We should review if it should
        // be put behind a mut reference.
        #[namespace = "mlx_cxx"]
        fn eval(outputs: &[UniquePtr<array>]);

        // // TODO: this needs to be placed in a separate file
        // #[namespace = "mlx_cxx"]
        // fn execute_callback(f: &DynFn, args: i32) -> i32;

        #[namespace = "mlx_cxx"]
        #[rust_name = "vjp_multiary_cxx_fn"]
        fn vjp(f: &MultiaryCxxFn, primals: &[UniquePtr<array>], cotangents: &[UniquePtr<array>]) -> [UniquePtr<CxxVector<array>>; 2];

        #[namespace = "mlx_cxx"]
        #[rust_name = "vjp_unary_cxx_fn"]
        fn vjp(f: &UnaryCxxFn, primal: &array, cotangent: &array) -> [UniquePtr<array>; 2];

        #[namespace = "mlx_cxx"]
        #[rust_name = "jvp_multiary_cxx_fn"]
        fn jvp(f: &MultiaryCxxFn, primals: &[UniquePtr<array>], tangents: &[UniquePtr<array>]) -> [UniquePtr<CxxVector<array>>; 2];

        #[namespace = "mlx_cxx"]
        #[rust_name = "jvp_unary_cxx_fn"]
        fn jvp(f: &UnaryCxxFn, primal: &array, tangent: &array) -> [UniquePtr<array>; 2];
    }
}
