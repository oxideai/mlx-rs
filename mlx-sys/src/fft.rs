#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx-cxx/utils.hpp");
        include!("mlx-cxx/fft.hpp");

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        #[namespace = "mlx_cxx"]
        type StreamOrDevice = crate::utils::StreamOrDevice;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn_shape_axes"]
        fn fftn(
            a: &array,
            n: &CxxVector<i32>,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn_axes"]
        fn fftn(
            a: &array,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn"]
        fn fftn(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn_shape_axes"]
        fn ifftn(
            a: &array,
            n: &CxxVector<i32>,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn_axes"]
        fn ifftn(
            a: &array,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn"]
        fn ifftn(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft_shape_axis"]
        fn fft(
            a: &array,
            n: i32,
            axis: i32,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft_axis"]
        fn fft(a: &array, axis: i32, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft"]
        fn fft(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft_shape_axis"]
        fn ifft(
            a: &array,
            n: i32,
            axis: i32,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft_axis"]
        fn ifft(a: &array, axis: i32, stream_or_device: StreamOrDevice)
            -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft"]
        fn ifft(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2_shape_axes"]
        fn fft2(
            a: &array,
            n: &CxxVector<i32>,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2_axes"]
        fn fft2(
            a: &array,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2"]
        fn fft2(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2_shape_axes"]
        fn ifft2(
            a: &array,
            n: &CxxVector<i32>,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2_axes"]
        fn ifft2(
            a: &array,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2"]
        fn ifft2(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn_shape_axes"]
        fn rfftn(
            a: &array,
            n: &CxxVector<i32>,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn_axes"]
        fn rfftn(
            a: &array,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn"]
        fn rfftn(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn_shape_axes"]
        fn irfftn(
            a: &array,
            n: &CxxVector<i32>,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn_axes"]
        fn irfftn(
            a: &array,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn"]
        fn irfftn(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft_shape_axis"]
        fn rfft(
            a: &array,
            n: i32,
            axis: i32,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft_axis"]
        fn rfft(a: &array, axis: i32, stream_or_device: StreamOrDevice)
            -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft"]
        fn rfft(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft_shape_axis"]
        fn irfft(
            a: &array,
            n: i32,
            axis: i32,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft_axis"]
        fn irfft(
            a: &array,
            axis: i32,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft"]
        fn irfft(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2_shape_axes"]
        fn rfft2(
            a: &array,
            n: &CxxVector<i32>,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2_axes"]
        fn rfft2(
            a: &array,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2"]
        fn rfft2(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2_shape_axes"]
        fn irfft2(
            a: &array,
            n: &CxxVector<i32>,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2_axes"]
        fn irfft2(
            a: &array,
            axes: &CxxVector<i32>,
            stream_or_device: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2"]
        fn irfft2(a: &array, stream_or_device: StreamOrDevice) -> Result<UniquePtr<array>>;
    }
}
