#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx-cxx/ops.hpp");

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        #[namespace = "mlx_cxx"]
        type StreamOrDevice = crate::StreamOrDevice;

        #[namespace = "mlx::core"]
        type Dtype = crate::dtype::ffi::Dtype;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_f64"]
        fn arange(
            start: f64,
            stop: f64,
            step: f64,
            stream_or_device: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_start_stop_dtype"]
        fn arange(
            start: f64,
            stop: f64,
            dtype: Dtype,
            stream_or_device: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_start_stop_f64"]
        fn arange(start: f64, stop: f64, stream_or_device: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_stop_dtype"]
        fn arange(stop: f64, dtype: Dtype, stream_or_device: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_stop_f64"]
        fn arange(stop: f64, stream_or_device: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_i32"]
        fn arange(
            start: i32,
            stop: i32,
            step: i32,
            stream_or_device: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_start_stop_i32"]
        fn arange(start: i32, stop: i32, stream_or_device: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_stop_i32"]
        fn arange(stop: i32, stream_or_device: StreamOrDevice) -> UniquePtr<array>;
    }
}

#[cfg(test)]
mod tests {
    use super::ffi;

    #[test]
    fn test_arange() {
        let array = ffi::arange_f64(0.0, 10.0, 1.0, Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_i32(0, 10, 1, Default::default());
        assert_eq!(array.size(), 10);
    }

    #[test]
    fn test_arange_start_stop() {
        let array = ffi::arange_start_stop_dtype(0.0, 10.0, crate::dtype::ffi::dtype_float32(), Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_start_stop_dtype(0.0, 10.0, crate::dtype::ffi::dtype_int32(), Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_start_stop_f64(0.0, 10.0, Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_start_stop_i32(0, 10, Default::default());
        assert_eq!(array.size(), 10);
    }

    #[test]
    fn test_arange_stop() {
        let array = ffi::arange_stop_dtype(10.0, crate::dtype::ffi::dtype_float32(), Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_stop_dtype(10.0, crate::dtype::ffi::dtype_int32(), Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_stop_f64(10.0, Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_stop_i32(10, Default::default());
        assert_eq!(array.size(), 10);
    }


}
