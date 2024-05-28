#[macro_export]
macro_rules! fft {
    ($a:expr) => {
        $crate::fft::fft($a, None, None)
    };
    ($a:expr, $n:expr) => {
        $crate::fft::fft($a, $n, None)
    };
    ($a:expr, $n:expr, $axis:expr) => {
        $crate::fft::fft($a, $n, $axis)
    };
    ($a:expr, $n:expr, $axis:expr, $stream:expr) => {
        $crate::fft::fft_device($a, $n, $axis, $stream)
    };
}

#[macro_export]
macro_rules! fft2 {
    ($a:expr) => {
        $crate::fft::fft2($a, None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::fft2($a, $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::fft2($a, $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, $stream:expr) => {
        $crate::fft::fft2_device($a, $shape, $axes, $stream)
    };
}

#[macro_export]
macro_rules! fftn {
    ($a:expr) => {
        $crate::fft::fftn($a, None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::fftn($a, $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::fftn($a, $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, $stream:expr) => {
        $crate::fft::fftn_device($a, $shape, $axes, $stream)
    };
}

#[macro_export]
macro_rules! ifft {
    ($a:expr) => {
        $crate::fft::ifft($a, None, None)
    };
    ($a:expr, $n:expr) => {
        $crate::fft::ifft($a, $n, None)
    };
    ($a:expr, $n:expr, $axis:expr) => {
        $crate::fft::ifft($a, $n, $axis)
    };
    ($a:expr, $n:expr, $axis:expr, $stream:expr) => {
        $crate::fft::ifft_device($a, $n, $axis, $stream)
    };
}

#[macro_export]
macro_rules! ifft2 {
    ($a:expr) => {
        $crate::fft::ifft2($a, None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::ifft2($a, $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::ifft2($a, $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, $stream:expr) => {
        $crate::fft::ifft2_device($a, $shape, $axes, $stream)
    };
}

#[macro_export]
macro_rules! ifftn {
    ($a:expr) => {
        $crate::fft::ifftn($a, None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::ifftn($a, $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::ifftn($a, $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, $stream:expr) => {
        $crate::fft::ifftn_device($a, $shape, $axes, $stream)
    };
}

#[macro_export]
macro_rules! rfft {
    ($a:expr) => {
        $crate::fft::rfft($a, None, None)
    };
    ($a:expr, $n:expr) => {
        $crate::fft::rfft($a, $n, None)
    };
    ($a:expr, $n:expr, $axis:expr) => {
        $crate::fft::rfft($a, $n, $axis)
    };
    ($a:expr, $n:expr, $axis:expr, $stream:expr) => {
        $crate::fft::rfft_device($a, $n, $axis, $stream)
    };
}

#[macro_export]
macro_rules! rfft2 {
    ($a:expr) => {
        $crate::fft::rfft2($a, None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::rfft2($a, $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::rfft2($a, $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, $stream:expr) => {
        $crate::fft::rfft2_device($a, $shape, $axes, $stream)
    };
}

#[macro_export]
macro_rules! rfftn {
    ($a:expr) => {
        $crate::fft::rfftn($a, None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::rfftn($a, $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::rfftn($a, $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, $stream:expr) => {
        $crate::fft::rfftn_device($a, $shape, $axes, $stream)
    };
}

#[macro_export]
macro_rules! irfft {
    ($a:expr) => {
        $crate::fft::irfft($a, None, None)
    };
    ($a:expr, $n:expr) => {
        $crate::fft::irfft($a, $n, None)
    };
    ($a:expr, $n:expr, $axis:expr) => {
        $crate::fft::irfft($a, $n, $axis)
    };
    ($a:expr, $n:expr, $axis:expr, $stream:expr) => {
        $crate::fft::irfft_device($a, $n, $axis, $stream)
    };
}

#[macro_export]
macro_rules! irfft2 {
    ($a:expr) => {
        $crate::fft::irfft2($a, None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::irfft2($a, $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::irfft2($a, $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, $stream:expr) => {
        $crate::fft::irfft2_device($a, $shape, $axes, $stream)
    };
}

#[macro_export]
macro_rules! irfftn {
    ($a:expr) => {
        $crate::fft::irfftn($a, None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::irfftn($a, $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::irfftn($a, $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, $stream:expr) => {
        $crate::fft::irfftn_device($a, $shape, $axes, $stream)
    };
}

#[cfg(test)]
mod tests {
    use crate::{Array, StreamOrDevice};

    #[test]
    fn test_fft() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let fft_a = fft!(&a);
        let ifft_a = ifft!(&fft_a);
        assert_eq!(a, ifft_a);

        let fft_a = fft!(&a, 4);
        let ifft_a = ifft!(&fft_a, 4);
        assert_eq!(a, ifft_a);

        let fft_a = fft!(&a, 4, 0);
        let ifft_a = ifft!(&fft_a, 4, 0);
        assert_eq!(a, ifft_a);

        let fft_a = fft!(&a, 4, 0, StreamOrDevice::cpu());
        let ifft_a = ifft!(&fft_a, 4, 0, StreamOrDevice::cpu());
        assert_eq!(a, ifft_a);
    }

    #[test]
    fn test_fft2() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let fft2_a = fft2!(&a);
        let ifft2_a = ifft2!(&fft2_a);
        assert_eq!(a, ifft2_a);

        let fft2_a = fft2!(&a, &[2, 2][..]);
        let ifft2_a = ifft2!(&fft2_a, &[2, 2][..]);
        assert_eq!(a, ifft2_a);

        let fft2_a = fft2!(&a, &[2, 2][..], &[0, 1][..]);
        let ifft2_a = ifft2!(&fft2_a, &[2, 2][..], &[0, 1][..]);
        assert_eq!(a, ifft2_a);

        let fft2_a = fft2!(&a, &[2, 2][..], &[0, 1][..], StreamOrDevice::cpu());
        let ifft2_a = ifft2!(&fft2_a, &[2, 2][..], &[0, 1][..], StreamOrDevice::cpu());
        assert_eq!(a, ifft2_a);
    }

    #[test]
    fn test_fftn() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let fftn_a = fftn!(&a);
        let ifftn_a = ifftn!(&fftn_a);
        assert_eq!(a, ifftn_a);

        let fftn_a = fftn!(&a, &[2, 2][..]);
        let ifftn_a = ifftn!(&fftn_a, &[2, 2][..]);
        assert_eq!(a, ifftn_a);

        let fftn_a = fftn!(&a, &[2, 2][..], &[0, 1][..]);
        let ifftn_a = ifftn!(&fftn_a, &[2, 2][..], &[0, 1][..]);
        assert_eq!(a, ifftn_a);

        let fftn_a = fftn!(&a, &[2, 2][..], &[0, 1][..], StreamOrDevice::cpu());
        let ifftn_a = ifftn!(&fftn_a, &[2, 2][..], &[0, 1][..], StreamOrDevice::cpu());
        assert_eq!(a, ifftn_a);
    }

    #[test]
    fn test_rfft() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let rfft_a = rfft!(&a);
        let irfft_a = irfft!(&rfft_a);
        assert_eq!(a, irfft_a);

        let rfft_a = rfft!(&a, 4);
        let irfft_a = irfft!(&rfft_a, 4);
        assert_eq!(a, irfft_a);

        let rfft_a = rfft!(&a, 4, 0);
        let irfft_a = irfft!(&rfft_a, 4, 0);
        assert_eq!(a, irfft_a);

        let rfft_a = rfft!(&a, 4, 0, StreamOrDevice::cpu());
        let irfft_a = irfft!(&rfft_a, 4, 0, StreamOrDevice::cpu());
        assert_eq!(a, irfft_a);
    }

    #[test]
    fn test_rfft2() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let rfft2_a = rfft2!(&a);
        let irfft2_a = irfft2!(&rfft2_a);
        assert_eq!(a, irfft2_a);

        let rfft2_a = rfft2!(&a, &[2, 2][..]);
        let irfft2_a = irfft2!(&rfft2_a, &[2, 2][..]);
        assert_eq!(a, irfft2_a);

        let rfft2_a = rfft2!(&a, &[2, 2][..], &[0, 1][..]);
        let irfft2_a = irfft2!(&rfft2_a, &[2, 2][..], &[0, 1][..]);
        assert_eq!(a, irfft2_a);

        let rfft2_a = rfft2!(&a, &[2, 2][..], &[0, 1][..], StreamOrDevice::cpu());
        let irfft2_a = irfft2!(&rfft2_a, &[2, 2][..], &[0, 1][..], StreamOrDevice::cpu());
        assert_eq!(a, irfft2_a);
    }

    #[test]
    fn test_rfftn() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let rfftn_a = rfftn!(&a);
        let irfftn_a = irfftn!(&rfftn_a);
        assert_eq!(a, irfftn_a);

        let rfftn_a = rfftn!(&a, &[2, 2][..]);
        let irfftn_a = irfftn!(&rfftn_a, &[2, 2][..]);
        assert_eq!(a, irfftn_a);

        let rfftn_a = rfftn!(&a, &[2, 2][..], &[0, 1][..]);
        let irfftn_a = irfftn!(&rfftn_a, &[2, 2][..], &[0, 1][..]);
        assert_eq!(a, irfftn_a);

        let rfftn_a = rfftn!(&a, &[2, 2][..], &[0, 1][..], StreamOrDevice::cpu());
        let irfftn_a = irfftn!(&rfftn_a, &[2, 2][..], &[0, 1][..], StreamOrDevice::cpu());
        assert_eq!(a, irfftn_a);
    }
}
