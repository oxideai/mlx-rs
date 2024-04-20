use mlx_macros::default_device;

use crate::{array::Array, stream::StreamOrDevice};

/// One dimensional discrete Fourier Transform.
///
/// # Params
///
/// - a: The input array.
/// - n: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]`.
/// - axis: Axis along which to perform the FFT. The default is -1.
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
pub unsafe fn fft_device_unchecked(
    a: Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    let axis = axis.into().unwrap_or(-1);
    let n = n.into().unwrap_or_else(|| {
        if axis.is_negative() {
            // TODO: replace with unchecked_add when it's stable
            let index = (a.ndim() as i32)
                .checked_add(axis)
                .unwrap()
                // index may still be negative
                .max(0) as usize;
            a.shape()[index]
        } else {
            // # Safety: positive i32 is always smaller than usize::MAX
            a.shape()[axis as usize]
        }
    });
    unsafe {
        let c_array = mlx_sys::mlx_fft_fft(a.c_array, n, axis, stream.stream.c_stream);
        Array::from_ptr(c_array)
    }
}

#[cfg(test)]
mod tests {
    use crate::{array::complex64, stream::StreamOrDevice};

    use super::*;

    #[test]
    fn test_fft_unchecked() {
        let array = crate::array::Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let mut result = unsafe { fft_unchecked(array, 4, 0) };
        result.eval();

        assert_eq!(result.dtype(), crate::dtype::Dtype::Complex64);

        let expected = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);
    }

    #[test]
    fn test_fft_device_unchecked() {
        let array = crate::array::Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let s = StreamOrDevice::cpu();
        let mut result = unsafe { fft_device_unchecked(array, 4, 0, s) };
        result.eval();

        assert_eq!(result.dtype(), crate::dtype::Dtype::Complex64);

        let expected = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);
    }
}
