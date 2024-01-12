#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("mlx/utils.h");
        include!("mlx-cxx/utils.hpp");

        #[namespace = "mlx::core"]
        type Dtype = crate::dtype::ffi::Dtype;

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        #[namespace = "mlx::core"]
        fn result_type(arrays: &CxxVector<array>) -> Dtype;

        #[namespace = "mlx_cxx"]
        fn broadcast_shapes(s1: &CxxVector<i32>, s2: &CxxVector<i32>) -> UniquePtr<CxxVector<i32>>;

        #[namespace = "mlx::core"]
        fn is_same_shape(arrays: &CxxVector<array>) -> bool;

        #[namespace = "mlx::core"]
        fn normalize_axis(axis: i32, ndim: i32) -> i32;
    }
}