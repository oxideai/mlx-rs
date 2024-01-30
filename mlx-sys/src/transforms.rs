//! The function pointer used in functions like `grad()` are used to
//! eventually compute the output `array` which is then used to compute the gradient.
//! So theoretically we could instead pass a rust function that is callable from C++.

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/array.h");
        include!("mlx-cxx/transforms.hpp");

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        #[namespace = "mlx_cxx"]
        type CxxUnaryFn;

        #[namespace = "mlx_cxx"]
        type CxxMultiaryFn;

        #[namespace = "mlx_cxx"]
        type CxxMultiInputSingleOutputFn;

        #[namespace = "mlx_cxx"]
        type CxxPairInputSingleOutputFn;

        #[namespace = "mlx_cxx"]
        type CxxSingleInputPairOutputFn;

        #[namespace = "mlx::core"]
        #[cxx_name = "ValueAndGradFn"]
        type CxxValueAndGradFn;

        #[namespace = "mlx::core"]
        #[cxx_name = "SimpleValueAndGradFn"]
        type CxxSimpleValueAndGradFn;

        // TODO: This clearly changes internal states of the arrays. We should review if it should
        // be put behind a mut reference.
        #[namespace = "mlx_cxx"]
        fn simplify(outputs: &[UniquePtr<array>]);

        // TODO: This clearly changes internal states of the arrays. We should review if it should
        // be put behind a mut reference.
        #[namespace = "mlx_cxx"]
        fn eval(outputs: &[UniquePtr<array>]) -> Result<()>;

        // // TODO: this needs to be placed in a separate file
        // #[namespace = "mlx_cxx"]
        // fn execute_callback(f: &DynFn, args: i32) -> i32;

        #[namespace = "mlx_cxx"]
        #[rust_name = "vjp_multiary_cxx_fn"]
        fn vjp(f: &CxxMultiaryFn, primals: &[UniquePtr<array>], cotangents: &[UniquePtr<array>]) -> Result<[UniquePtr<CxxVector<array>>; 2]>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "vjp_unary_cxx_fn"]
        fn vjp(f: &CxxUnaryFn, primal: &array, cotangent: &array) -> Result<[UniquePtr<array>; 2]>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "jvp_multiary_cxx_fn"]
        fn jvp(f: &CxxMultiaryFn, primals: &[UniquePtr<array>], tangents: &[UniquePtr<array>]) -> Result<[UniquePtr<CxxVector<array>>; 2]>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "jvp_unary_cxx_fn"]
        fn jvp(f: &CxxUnaryFn, primal: &array, tangent: &array) -> Result<[UniquePtr<array>; 2]>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "value_and_grad_multiary_cxx_fn_argnums"]
        fn value_and_grad(f: &CxxMultiaryFn, argnums: &CxxVector<i32>) -> Result<UniquePtr<CxxValueAndGradFn>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "value_and_grad_multiary_cxx_fn_argnum"]
        fn value_and_grad(f: &CxxMultiaryFn, argnum: i32) -> Result<UniquePtr<CxxValueAndGradFn>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "value_and_grad_unary_cxx_fn"]
        fn value_and_grad(f: &CxxUnaryFn) -> Result<UniquePtr<CxxSingleInputPairOutputFn>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "value_and_grad_multi_input_single_output_cxx_fn"]
        fn value_and_grad(f: &CxxMultiInputSingleOutputFn, argnums: &CxxVector<i32>) -> Result<UniquePtr<CxxSimpleValueAndGradFn>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "grad_multi_input_single_output_cxx_fn_argnums"]
        fn grad(f: &CxxMultiInputSingleOutputFn, argnums: &CxxVector<i32>) -> Result<UniquePtr<CxxMultiaryFn>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "grad_multi_input_single_output_cxx_fn_argnum"]
        fn grad(f: &CxxMultiInputSingleOutputFn, argnum: i32) -> Result<UniquePtr<CxxMultiaryFn>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "grad_unary_cxx_fn"]
        fn grad(f: &CxxUnaryFn) -> Result<UniquePtr<CxxUnaryFn>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "vmap_unary_cxx_fn"]
        fn vmap(f: &CxxUnaryFn, in_axis: i32, out_axis: i32) -> Result<UniquePtr<CxxUnaryFn>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "vmap_pair_input_single_output_cxx_fn"]
        fn vmap(f: &CxxPairInputSingleOutputFn, in_axis_a: i32, in_axis_b: i32, out_axis: i32) -> Result<UniquePtr<CxxPairInputSingleOutputFn>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "vmap_multiary_cxx_fn"]
        fn vmap(f: &CxxMultiaryFn, in_axes: &CxxVector<i32>, out_axes: &CxxVector<i32>) -> Result<UniquePtr<CxxMultiaryFn>>;
    }
}
