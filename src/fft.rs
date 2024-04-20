use crate::{array::Array, stream::StreamOrDevice};

pub fn fft(array: Array, n: i32, axis: i32) -> Array {
    // FFT is not yet implemented on gpu
    let s = StreamOrDevice::cpu();
    unsafe {
        let c_array = mlx_sys::mlx_fft_fft(array.c_array, n, axis, s.stream.c_stream);
        Array::from_ptr(c_array)
    }
}

pub fn fft_device(array: Array, n: i32, axis: i32, s: StreamOrDevice) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_fft_fft(array.c_array, n, axis, s.stream.c_stream);
        Array::from_ptr(c_array)
    }
}

#[cfg(test)]
mod tests {
    use crate::{array::complex64, fft::fft_device, stream::StreamOrDevice};

    use super::fft;

    #[test]
    fn test_fft() {
        let array = crate::array::Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let mut result = fft(array, 4, 0);
        result.eval();

        assert_eq!(result.dtype(), crate::dtype::Dtype::Complex64);

        let expected = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), Some(&expected[..]));
    }

    #[test]
    fn test_fft_device() {
        let array = crate::array::Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let s = StreamOrDevice::cpu();
        let mut result = fft_device(array, 4, 0, s);
        result.eval();

        assert_eq!(result.dtype(), crate::dtype::Dtype::Complex64);

        let expected = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), Some(&expected[..]));
    }
}