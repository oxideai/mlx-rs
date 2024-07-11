#[macro_export]
macro_rules! abs {
    ($a:expr) => {
        $a.abs()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.abs_device($stream)
    };
}

#[macro_export]
macro_rules! add {
    ($a:expr, $b:expr) => {
        $a.add_device($b.as_ref(), $crate::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.add_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! sub {
    ($a:expr, $b:expr) => {
        $a.subtract_device($b.as_ref(), $crate::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.subtract_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! neg {
    ($a:expr) => {
        $a.negative()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.negative_device($stream)
    };
}

#[macro_export]
macro_rules! logical_not {
    ($a:expr) => {
        $a.logical_not()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.logical_not_device($stream)
    };
}

#[macro_export]
macro_rules! mul {
    ($a:expr, $b:expr) => {
        $a.multiply_device($b.as_ref(), $crate::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.multiply_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! div {
    ($a:expr, $b:expr) => {
        $a.divide_device($b.as_ref(), $crate::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.divide_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! pow {
    ($a:expr, $b:expr) => {
        $a.power_device($b.as_ref(), $crate::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.power_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! rem {
    ($a:expr, $b:expr) => {
        $a.remainder_device($b.as_ref(), $crate::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.remainder_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! sqrt {
    ($a:expr) => {
        $a.sqrt()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.sqrt_device($stream)
    };
}

#[macro_export]
macro_rules! cos {
    ($a:expr) => {
        $a.cos()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.cos_device($stream)
    };
}

#[macro_export]
macro_rules! exp {
    ($a:expr) => {
        $a.exp()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.exp_device($stream)
    };
}

#[macro_export]
macro_rules! floor {
    ($a:expr) => {
        $a.floor()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.floor_device($stream)
    };
}

#[macro_export]
macro_rules! log {
    ($a:expr) => {
        $a.log()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.log_device($stream)
    };
}

#[macro_export]
macro_rules! log2 {
    ($a:expr) => {
        $a.log2()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.log2_device($stream)
    };
}

#[macro_export]
macro_rules! log10 {
    ($a:expr) => {
        $a.log10()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.log10_device($stream)
    };
}

#[macro_export]
macro_rules! log1p {
    ($a:expr) => {
        $a.log1p()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.log1p_device($stream)
    };
}

#[macro_export]
macro_rules! matmul {
    ($a:expr, $b:expr) => {
        $a.matmul($b.as_ref())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.matmul_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! reciprocal {
    ($a:expr) => {
        $a.reciprocal()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.reciprocal_device($stream)
    };
}

#[macro_export]
macro_rules! round {
    ($a:expr) => {
        $a.round(None)
    };
    ($a:expr, stream=$stream:expr) => {
        $a.round_device(None, $stream)
    };
    ($a:expr, $decimals:expr) => {
        $a.round($decimals)
    };
    ($a:expr, $decimals:expr, stream=$stream:expr) => {
        $a.round_device($decimals, $stream)
    };
}

#[macro_export]
macro_rules! rsqrt {
    ($a:expr) => {
        $a.rsqrt()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.rsqrt_device($stream)
    };
}

#[macro_export]
macro_rules! sin {
    ($a:expr) => {
        $a.sin()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.sin_device($stream)
    };
}

#[macro_export]
macro_rules! square {
    ($a:expr) => {
        $a.square()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.square_device($stream)
    };
}

#[cfg(test)]
mod tests {
    use crate::{Array, StreamOrDevice};

    #[test]
    fn test_abs() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[-1, -2, -3], &[3]);

        // We are just testing that the macro compiles
        let _ = abs!(a);
        let _ = abs!(a, stream = stream);
    }

    #[test]
    fn test_add() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        // We are just testing that the macro compiles
        let _ = add!(a, b);
        let _ = add!(a, b, stream = stream);
    }

    #[test]
    fn test_sub() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        // We are just testing that the macro compiles
        let _ = sub!(a, b);
        let _ = sub!(a, b, stream = stream);
    }

    #[test]
    fn test_neg() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = neg!(a);
        let _ = neg!(a, stream = stream);
    }

    #[test]
    fn test_logical_not() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 0, 1], &[3]);

        // We are just testing that the macro compiles
        let _ = logical_not!(a);
        let _ = logical_not!(a, stream = stream);
    }

    #[test]
    fn test_mul() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        // We are just testing that the macro compiles
        let _ = mul!(a, b);
        let _ = mul!(a, b, stream = stream);
    }

    #[test]
    fn test_div() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        // We are just testing that the macro compiles
        let _ = div!(a, b);
        let _ = div!(a, b, stream = stream);
    }

    #[test]
    fn test_pow() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        // We are just testing that the macro compiles
        let _ = pow!(a, b);
        let _ = pow!(a, b, stream = stream);
    }

    #[test]
    fn test_rem() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        // We are just testing that the macro compiles
        let _ = rem!(a, b);
        let _ = rem!(a, b, stream = stream);
    }

    #[test]
    fn test_sqrt() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = sqrt!(a);
        let _ = sqrt!(a, stream = stream);
    }

    #[test]
    fn test_cos() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = cos!(a);
        let _ = cos!(a, stream = stream);
    }

    #[test]
    fn test_exp() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = exp!(a);
        let _ = exp!(a, stream = stream);
    }

    #[test]
    fn test_floor() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.1, 2.2, 3.3], &[3]);

        // We are just testing that the macro compiles
        let _ = floor!(a);
        let _ = floor!(a, stream = stream);
    }

    #[test]
    fn test_log() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = log!(a);
        let _ = log!(a, stream = stream);
    }

    #[test]
    fn test_log2() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = log2!(a);
        let _ = log2!(a, stream = stream);
    }

    #[test]
    fn test_log10() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = log10!(a);
        let _ = log10!(a, stream = stream);
    }

    #[test]
    fn test_log1p() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = log1p!(a);
        let _ = log1p!(a, stream = stream);
    }

    #[test]
    fn test_matmul() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Array::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

        // We are just testing that the macro compiles
        let _ = matmul!(a, b);
        let _ = matmul!(a, b, stream = stream);
    }

    #[test]
    fn test_reciprocal() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = reciprocal!(a);
        let _ = reciprocal!(a, stream = stream);
    }

    #[test]
    fn test_round() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.1, 2.2, 3.3], &[3]);

        // We are just testing that the macro compiles
        let _ = round!(a);
        let _ = round!(a, 1);
        let _ = round!(a, stream = stream);
    }

    #[test]
    fn test_rsqrt() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = rsqrt!(a);
        let _ = rsqrt!(a, stream = stream);
    }

    #[test]
    fn test_sin() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = sin!(a);
        let _ = sin!(a, stream = stream);
    }

    #[test]
    fn test_square() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = square!(a);
        let _ = square!(a, stream = stream);
    }
}
