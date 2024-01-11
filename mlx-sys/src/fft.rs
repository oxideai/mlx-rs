#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx-cxx/fft.hpp");

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        #[namespace = "mlx::core"]
        type Stream = crate::stream::ffi::Stream;

        #[namespace = "mlx::core"]
        type Device = crate::device::ffi::Device;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn_shape_axes"]
        fn fftn_default(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn_shape_axes_stream"]
        fn fftn_stream(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn_shape_axes_device"]
        fn fftn_device(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn_axes"]
        fn fftn_default(a: &array, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn_axes_stream"]
        fn fftn_stream(a: &array, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn_axes_device"]
        fn fftn_device(a: &array, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn"]
        fn fftn_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn_stream"]
        fn fftn_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fftn_device"]
        fn fftn_device(a: &array, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn_shape_axes"]
        fn ifftn_default(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn_shape_axes_stream"]
        fn ifftn_stream(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn_shape_axes_device"]
        fn ifftn_device(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn_axes"]
        fn ifftn_default(a: &array, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn_axes_stream"]
        fn ifftn_stream(a: &array, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn_axes_device"]
        fn ifftn_device(a: &array, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn"]
        fn ifftn_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn_stream"]
        fn ifftn_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifftn_device"]
        fn ifftn_device(a: &array, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft_shape_axis"]
        fn fft_default(a: &array, n: i32, axis: i32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft_shape_axis_stream"]
        fn fft_stream(a: &array, n: i32, axis: i32, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft_shape_axis_device"]
        fn fft_device(a: &array, n: i32, axis: i32, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft_axis"]
        fn fft_default(a: &array, axis: i32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft_axis_stream"]
        fn fft_stream(a: &array, axis: i32, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft_axis_device"]
        fn fft_device(a: &array, axis: i32, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft"]
        fn fft_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft_stream"]
        fn fft_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft_device"]
        fn fft_device(a: &array, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft_shape_axis"]
        fn ifft_default(a: &array, n: i32, axis: i32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft_shape_axis_stream"]
        fn ifft_stream(a: &array, n: i32, axis: i32, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft_shape_axis_device"]
        fn ifft_device(a: &array, n: i32, axis: i32, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft_axis"]
        fn ifft_default(a: &array, axis: i32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft_axis_stream"]
        fn ifft_stream(a: &array, axis: i32, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft_axis_device"]
        fn ifft_device(a: &array, axis: i32, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft"]
        fn ifft_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft_stream"]
        fn ifft_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft_device"]
        fn ifft_device(a: &array, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2_shape_axes"]
        fn fft2_default(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2_shape_axes_stream"]
        fn fft2_stream(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2_shape_axes_device"]
        fn fft2_device(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2_axes"]
        fn fft2_default(a: &array, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2_axes_stream"]
        fn fft2_stream(a: &array, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2_axes_device"]
        fn fft2_device(a: &array, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2"]
        fn fft2_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2_stream"]
        fn fft2_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "fft2_device"]
        fn fft2_device(a: &array, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2_shape_axes"]
        fn ifft2_default(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2_shape_axes_stream"]
        fn ifft2_stream(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2_shape_axes_device"]
        fn ifft2_device(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2_axes"]
        fn ifft2_default(a: &array, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2_axes_stream"]
        fn ifft2_stream(a: &array, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2_axes_device"]
        fn ifft2_device(a: &array, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2"]
        fn ifft2_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2_stream"]
        fn ifft2_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ifft2_device"]
        fn ifft2_device(a: &array, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn_shape_axes"]
        fn rfftn_default(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn_shape_axes_stream"]
        fn rfftn_stream(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn_shape_axes_device"]
        fn rfftn_device(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn_axes"]
        fn rfftn_default(a: &array, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn_axes_stream"]
        fn rfftn_stream(a: &array, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn_axes_device"]
        fn rfftn_device(a: &array, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn"]
        fn rfftn_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn_stream"]
        fn rfftn_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfftn_device"]
        fn rfftn_device(a: &array, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn_shape_axes"]
        fn irfftn_default(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn_shape_axes_stream"]
        fn irfftn_stream(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn_shape_axes_device"]
        fn irfftn_device(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn_axes"]
        fn irfftn_default(a: &array, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn_axes_stream"]
        fn irfftn_stream(a: &array, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn_axes_device"]
        fn irfftn_device(a: &array, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn"]
        fn irfftn_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn_stream"]
        fn irfftn_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfftn_device"]
        fn irfftn_device(a: &array, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft_shape_axis"]
        fn rfft_default(a: &array, n: i32, axis: i32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft_shape_axis_stream"]
        fn rfft_stream(a: &array, n: i32, axis: i32, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft_shape_axis_device"]
        fn rfft_device(a: &array, n: i32, axis: i32, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft_axis"]
        fn rfft_default(a: &array, axis: i32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft_axis_stream"]
        fn rfft_stream(a: &array, axis: i32, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft_axis_device"]
        fn rfft_device(a: &array, axis: i32, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft"]
        fn rfft_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft_stream"]
        fn rfft_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft_device"]
        fn rfft_device(a: &array, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft_shape_axis"]
        fn irfft_default(a: &array, n: i32, axis: i32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft_shape_axis_stream"]
        fn irfft_stream(a: &array, n: i32, axis: i32, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft_shape_axis_device"]
        fn irfft_device(a: &array, n: i32, axis: i32, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft_axis"]
        fn irfft_default(a: &array, axis: i32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft_axis_stream"]
        fn irfft_stream(a: &array, axis: i32, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft_axis_device"]
        fn irfft_device(a: &array, axis: i32, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft"]
        fn irfft_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft_stream"]
        fn irfft_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft_device"]
        fn irfft_device(a: &array, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2_shape_axes"]
        fn rfft2_default(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2_shape_axes_stream"]
        fn rfft2_stream(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2_shape_axes_device"]
        fn rfft2_device(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2_axes"]
        fn rfft2_default(a: &array, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2_axes_stream"]
        fn rfft2_stream(a: &array, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2_axes_device"]
        fn rfft2_device(a: &array, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2"]
        fn rfft2_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2_stream"]
        fn rfft2_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "rfft2_device"]
        fn rfft2_device(a: &array, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2_shape_axes"]
        fn irfft2_default(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2_shape_axes_stream"]
        fn irfft2_stream(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2_shape_axes_device"]
        fn irfft2_device(a: &array, n: &CxxVector<i32>, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2_axes"]
        fn irfft2_default(a: &array, axes: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2_axes_stream"]
        fn irfft2_stream(a: &array, axes: &CxxVector<i32>, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2_axes_device"]
        fn irfft2_device(a: &array, axes: &CxxVector<i32>, d: Device) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2"]
        fn irfft2_default(a: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2_stream"]
        fn irfft2_stream(a: &array, s: Stream) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "irfft2_device"]
        fn irfft2_device(a: &array, d: Device) -> UniquePtr<array>;
    }
}