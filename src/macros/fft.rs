/// See [`crate::fft::fft`] for details.
#[macro_export]
macro_rules! fft {
    ($a:expr) => {
        $crate::fft::fft($a.as_ref(), None, None)
    };
    ($a:expr, $n:expr) => {
        $crate::fft::fft($a.as_ref(), $n, None)
    };
    ($a:expr, $n:expr, $axis:expr) => {
        $crate::fft::fft($a.as_ref(), $n, $axis)
    };
    ($a:expr, $n:expr, $axis:expr, stream=$stream:expr) => {
        $crate::fft::fft_device($a.as_ref(), $n, $axis, $stream)
    };
}

/// See [`crate::fft::fft2`] for details.
#[macro_export]
macro_rules! fft2 {
    ($a:expr) => {
        $crate::fft::fft2($a.as_ref(), None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::fft2($a.as_ref(), $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::fft2($a.as_ref(), $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, stream=$stream:expr) => {
        $crate::fft::fft2_device($a.as_ref(), $shape, $axes, $stream)
    };
}

/// See [`crate::fft::fftn`] for details.
#[macro_export]
macro_rules! fftn {
    ($a:expr) => {
        $crate::fft::fftn($a.as_ref(), None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::fftn($a.as_ref(), $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::fftn($a.as_ref(), $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, stream=$stream:expr) => {
        $crate::fft::fftn_device($a.as_ref(), $shape, $axes, $stream)
    };
}

/// See [`crate::fft::ifft`] for details.
#[macro_export]
macro_rules! ifft {
    ($a:expr) => {
        $crate::fft::ifft($a.as_ref(), None, None)
    };
    ($a:expr, $n:expr) => {
        $crate::fft::ifft($a.as_ref(), $n, None)
    };
    ($a:expr, $n:expr, $axis:expr) => {
        $crate::fft::ifft($a.as_ref(), $n, $axis)
    };
    ($a:expr, $n:expr, $axis:expr, stream=$stream:expr) => {
        $crate::fft::ifft_device($a.as_ref(), $n, $axis, $stream)
    };
}

/// See [`crate::fft::ifft2`] for details.
#[macro_export]
macro_rules! ifft2 {
    ($a:expr) => {
        $crate::fft::ifft2($a.as_ref(), None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::ifft2($a.as_ref(), $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::ifft2($a.as_ref(), $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, stream=$stream:expr) => {
        $crate::fft::ifft2_device($a.as_ref(), $shape, $axes, $stream)
    };
}

/// See [`crate::fft::ifftn`] for details.
#[macro_export]
macro_rules! ifftn {
    ($a:expr) => {
        $crate::fft::ifftn($a.as_ref(), None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::ifftn($a.as_ref(), $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::ifftn($a.as_ref(), $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, stream=$stream:expr) => {
        $crate::fft::ifftn_device($a.as_ref(), $shape, $axes, $stream)
    };
}

/// See [`crate::fft::rfft`] for details.
#[macro_export]
macro_rules! rfft {
    ($a:expr) => {
        $crate::fft::rfft($a.as_ref(), None, None)
    };
    ($a:expr, $n:expr) => {
        $crate::fft::rfft($a.as_ref(), $n, None)
    };
    ($a:expr, $n:expr, $axis:expr) => {
        $crate::fft::rfft($a.as_ref(), $n, $axis)
    };
    ($a:expr, $n:expr, $axis:expr, stream=$stream:expr) => {
        $crate::fft::rfft_device($a.as_ref(), $n, $axis, $stream)
    };
}

/// See [`crate::fft::rfft2`] for details.
#[macro_export]
macro_rules! rfft2 {
    ($a:expr) => {
        $crate::fft::rfft2($a.as_ref(), None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::rfft2($a.as_ref(), $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::rfft2($a.as_ref(), $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, stream=$stream:expr) => {
        $crate::fft::rfft2_device($a.as_ref(), $shape, $axes, $stream)
    };
}

/// See [`crate::fft::rfftn`] for details.
#[macro_export]
macro_rules! rfftn {
    ($a:expr) => {
        $crate::fft::rfftn($a.as_ref(), None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::rfftn($a.as_ref(), $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::rfftn($a.as_ref(), $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, stream=$stream:expr) => {
        $crate::fft::rfftn_device($a.as_ref(), $shape, $axes, $stream)
    };
}

/// See [`crate::fft::irfft`] for details.
#[macro_export]
macro_rules! irfft {
    ($a:expr) => {
        $crate::fft::irfft($a.as_ref(), None, None)
    };
    ($a:expr, $n:expr) => {
        $crate::fft::irfft($a.as_ref(), $n, None)
    };
    ($a:expr, $n:expr, $axis:expr) => {
        $crate::fft::irfft($a.as_ref(), $n, $axis)
    };
    ($a:expr, $n:expr, $axis:expr, stream=$stream:expr) => {
        $crate::fft::irfft_device($a.as_ref(), $n, $axis, $stream)
    };
}

/// See [`crate::fft::irfft2`] for details.
#[macro_export]
macro_rules! irfft2 {
    ($a:expr) => {
        $crate::fft::irfft2($a.as_ref(), None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::irfft2($a.as_ref(), $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::irfft2($a.as_ref(), $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, stream=$stream:expr) => {
        $crate::fft::irfft2_device($a.as_ref(), $shape, $axes, $stream)
    };
}

/// See [`crate::fft::irfftn`] for details.
#[macro_export]
macro_rules! irfftn {
    ($a:expr) => {
        $crate::fft::irfftn($a.as_ref(), None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::fft::irfftn($a.as_ref(), $shape, None)
    };
    ($a:expr, $shape:expr, $axes:expr) => {
        $crate::fft::irfftn($a.as_ref(), $shape, $axes)
    };
    ($a:expr, $shape:expr, $axes:expr, stream=$stream:expr) => {
        $crate::fft::irfftn_device($a.as_ref(), $shape, $axes, $stream)
    };
}

#[cfg(test)]
mod tests {
    use crate::{Array, StreamOrDevice};

    #[test]
    fn test_fft() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let fft_a = fft!(&a).unwrap();
        let ifft_a = ifft!(&fft_a).unwrap();
        assert_eq!(a, ifft_a);

        let fft_a = fft!(&a, 4).unwrap();
        let ifft_a = ifft!(&fft_a, 4).unwrap();
        assert_eq!(a, ifft_a);

        let fft_a = fft!(&a, 4, 0).unwrap();
        let ifft_a = ifft!(&fft_a, 4, 0).unwrap();
        assert_eq!(a, ifft_a);

        let fft_a = fft!(&a, 4, 0, stream = StreamOrDevice::cpu()).unwrap();
        let ifft_a = ifft!(&fft_a, 4, 0, stream = StreamOrDevice::cpu()).unwrap();
        assert_eq!(a, ifft_a);
    }

    #[test]
    fn test_fft2() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let fft2_a = fft2!(&a).unwrap();
        let ifft2_a = ifft2!(&fft2_a).unwrap();
        assert_eq!(a, ifft2_a);

        let fft2_a = fft2!(&a, &[2, 2][..]).unwrap();
        let ifft2_a = ifft2!(&fft2_a, &[2, 2][..]).unwrap();
        assert_eq!(a, ifft2_a);

        let fft2_a = fft2!(&a, &[2, 2][..], &[0, 1][..]).unwrap();
        let ifft2_a = ifft2!(&fft2_a, &[2, 2][..], &[0, 1][..]).unwrap();
        assert_eq!(a, ifft2_a);

        let fft2_a = fft2!(&a, &[2, 2][..], &[0, 1][..], stream = StreamOrDevice::cpu()).unwrap();
        let ifft2_a = ifft2!(
            &fft2_a,
            &[2, 2][..],
            &[0, 1][..],
            stream = StreamOrDevice::cpu()
        )
        .unwrap();
        assert_eq!(a, ifft2_a);
    }

    #[test]
    fn test_fftn() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let fftn_a = fftn!(&a).unwrap();
        let ifftn_a = ifftn!(&fftn_a).unwrap();
        assert_eq!(a, ifftn_a);

        let fftn_a = fftn!(&a, &[2, 2][..]).unwrap();
        let ifftn_a = ifftn!(&fftn_a, &[2, 2][..]).unwrap();
        assert_eq!(a, ifftn_a);

        let fftn_a = fftn!(&a, &[2, 2][..], &[0, 1][..]).unwrap();
        let ifftn_a = ifftn!(&fftn_a, &[2, 2][..], &[0, 1][..]).unwrap();
        assert_eq!(a, ifftn_a);

        let fftn_a = fftn!(&a, &[2, 2][..], &[0, 1][..], stream = StreamOrDevice::cpu()).unwrap();
        let ifftn_a = ifftn!(
            &fftn_a,
            &[2, 2][..],
            &[0, 1][..],
            stream = StreamOrDevice::cpu()
        )
        .unwrap();
        assert_eq!(a, ifftn_a);
    }

    #[test]
    fn test_rfft() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let rfft_a = rfft!(&a).unwrap();
        let irfft_a = irfft!(&rfft_a).unwrap();
        assert_eq!(a, irfft_a);

        let rfft_a = rfft!(&a, 4).unwrap();
        let irfft_a = irfft!(&rfft_a, 4).unwrap();
        assert_eq!(a, irfft_a);

        let rfft_a = rfft!(&a, 4, 0).unwrap();
        let irfft_a = irfft!(&rfft_a, 4, 0).unwrap();
        assert_eq!(a, irfft_a);

        let rfft_a = rfft!(&a, 4, 0, stream = StreamOrDevice::cpu()).unwrap();
        let irfft_a = irfft!(&rfft_a, 4, 0, stream = StreamOrDevice::cpu()).unwrap();
        assert_eq!(a, irfft_a);
    }

    #[test]
    fn test_rfft2() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let rfft2_a = rfft2!(&a).unwrap();
        let irfft2_a = irfft2!(&rfft2_a).unwrap();
        assert_eq!(a, irfft2_a);

        let rfft2_a = rfft2!(&a, &[2, 2][..]).unwrap();
        let irfft2_a = irfft2!(&rfft2_a, &[2, 2][..]).unwrap();
        assert_eq!(a, irfft2_a);

        let rfft2_a = rfft2!(&a, &[2, 2][..], &[0, 1][..]).unwrap();
        let irfft2_a = irfft2!(&rfft2_a, &[2, 2][..], &[0, 1][..]).unwrap();
        assert_eq!(a, irfft2_a);

        let rfft2_a = rfft2!(&a, &[2, 2][..], &[0, 1][..], stream = StreamOrDevice::cpu()).unwrap();
        let irfft2_a = irfft2!(
            &rfft2_a,
            &[2, 2][..],
            &[0, 1][..],
            stream = StreamOrDevice::cpu()
        )
        .unwrap();
        assert_eq!(a, irfft2_a);
    }

    #[test]
    fn test_rfftn() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let rfftn_a = rfftn!(&a).unwrap();
        let irfftn_a = irfftn!(&rfftn_a).unwrap();
        assert_eq!(a, irfftn_a);

        let rfftn_a = rfftn!(&a, &[2, 2][..]).unwrap();
        let irfftn_a = irfftn!(&rfftn_a, &[2, 2][..]).unwrap();
        assert_eq!(a, irfftn_a);

        let rfftn_a = rfftn!(&a, &[2, 2][..], &[0, 1][..]).unwrap();
        let irfftn_a = irfftn!(&rfftn_a, &[2, 2][..], &[0, 1][..]).unwrap();
        assert_eq!(a, irfftn_a);

        let rfftn_a = rfftn!(&a, &[2, 2][..], &[0, 1][..], stream = StreamOrDevice::cpu()).unwrap();
        let irfftn_a = irfftn!(
            &rfftn_a,
            &[2, 2][..],
            &[0, 1][..],
            stream = StreamOrDevice::cpu()
        )
        .unwrap();
        assert_eq!(a, irfftn_a);
    }
}
