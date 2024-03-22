#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx-cxx/fast.hpp");

        #[namespace = "mlx_cxx"]
        type OptionalArray = crate::ops::ffi::OptionalArray;

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        #[namespace = "mlx_cxx"]
        type StreamOrDevice = crate::utils::StreamOrDevice;

        #[namespace = "mlx_cxx::fast"]
        fn rope(
            x: &array,
            dims: i32,
            traditional: bool,
            base: f32,
            scale: f32,
            offset: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx::fast"]
        fn scaled_dot_product_attention(
            queries: &array,
            keys: &array,
            values: &array,
            scale: f32,
            mask: &OptionalArray,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx::fast"]
        fn rms_norm(
            x: &array,
            weight: &array,
            eps: f32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx::fast"]
        fn layer_norm(
            x: &array,
            weight: &OptionalArray,
            bias: &OptionalArray,
            eps: f32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;
    }
}
