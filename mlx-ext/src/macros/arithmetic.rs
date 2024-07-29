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
macro_rules! acos {
    ($a:expr) => {
        $crate::mlx_rs::ops::acos($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::acos_device($a, $stream)
    };
}

#[macro_export]
macro_rules! acosh {
    ($a:expr) => {
        $crate::mlx_rs::ops::acosh($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::acosh_device($a, $stream)
    };
}

#[macro_export]
macro_rules! add {
    ($a:expr, $b:expr) => {
        $a.add_device($b.as_ref(), $crate::mlx_rs::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.add_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! asin {
    ($a:expr) => {
        $crate::mlx_rs::ops::asin($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::asin_device($a, $stream)
    };
}

#[macro_export]
macro_rules! asinh {
    ($a:expr) => {
        $crate::mlx_rs::ops::asinh($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::asinh_device($a, $stream)
    };
}

#[macro_export]
macro_rules! atan {
    ($a:expr) => {
        $crate::mlx_rs::ops::atan($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::atan_device($a, $stream)
    };
}

#[macro_export]
macro_rules! atanh {
    ($a:expr) => {
        $crate::mlx_rs::ops::atanh($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::atanh_device($a, $stream)
    };
}

#[macro_export]
macro_rules! ceil {
    ($a:expr) => {
        $crate::mlx_rs::ops::ceil($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::ceil_device($a, $stream)
    };
}

#[macro_export]
macro_rules! clip {
    ($a:expr, $bound:expr) => {
        $crate::mlx_rs::ops::clip($a, $bound)
    };
    ($a:expr, $bound:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::clip_device($a, $bound, $stream)
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
macro_rules! cosh {
    ($a:expr) => {
        $crate::mlx_rs::ops::cosh($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::cosh_device($a, $stream)
    };
}

#[macro_export]
macro_rules! degrees {
    ($a:expr) => {
        $crate::mlx_rs::ops::degrees($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::degrees_device($a, $stream)
    };
}

#[macro_export]
macro_rules! divide {
    ($a:expr, $b:expr) => {
        $a.divide_device($b.as_ref(), $crate::mlx_rs::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.divide_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! divmod {
    ($a:expr, $b:expr) => {
        $crate::mlx_rs::ops::divmod($a, $b)
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::divmod_device($a, $b, $stream)
    };
}

#[macro_export]
macro_rules! erf {
    ($a:expr) => {
        $crate::mlx_rs::ops::erf($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::erf_device($a, $stream)
    };
}

#[macro_export]
macro_rules! erfinv {
    ($a:expr) => {
        $crate::mlx_rs::ops::erfinv($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::erfinv_device($a, $stream)
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
macro_rules! expm1 {
    ($a:expr) => {
        $crate::mlx_rs::ops::expm1($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::expm1_device($a, $stream)
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
macro_rules! floor_divde {
    ($a:expr, $other:expr) => {
        $crate::mlx_rs::ops::floor_divide($a, $other)
    };
    ($a:expr, $other:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::floor_divide_device($a, $other, $stream)
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
macro_rules! log_add_exp {
    ($a:expr, $b:expr) => {
        $crate::mlx_rs::ops::log_add_exp($a, $b)
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::log_add_exp_device($a, $b, $stream)
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
macro_rules! maximum {
    ($a:expr, $b:expr) => {
        $crate::mlx_rs::ops::maximum($a, $b)
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::maximum_device($a, $b, $stream)
    };
}

#[macro_export]
macro_rules! minimum {
    ($a:expr, $b:expr) => {
        $crate::mlx_rs::ops::minimum($a, $b)
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::minimum_device($a, $b, $stream)
    };
}

#[macro_export]
macro_rules! multiply {
    ($a:expr, $b:expr) => {
        $a.multiply_device($b.as_ref(), $crate::mlx_rs::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.multiply_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! negative {
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
macro_rules! power {
    ($a:expr, $b:expr) => {
        $a.power_device($b.as_ref(), $crate::mlx_rs::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.power_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! radians {
    ($a:expr) => {
        $crate::mlx_rs::ops::radians($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::radians_device($a, $stream)
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
macro_rules! remainder {
    ($a:expr, $b:expr) => {
        $a.remainder_device($b.as_ref(), $crate::mlx_rs::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.remainder_device($b.as_ref(), $stream)
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
macro_rules! sigmoid {
    ($a:expr) => {
        $crate::mlx_rs::ops::sigmoid($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::sigmoid_device($a, $stream)
    };
}

#[macro_export]
macro_rules! sign {
    ($a:expr) => {
        $crate::mlx_rs::ops::sign($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::sign_device($a, $stream)
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
macro_rules! sinh {
    ($a:expr) => {
        $crate::mlx_rs::ops::sinh($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::sinh_device($a, $stream)
    };
}

#[macro_export]
macro_rules! softmax {
    ($a:expr) => {
        $crate::mlx_rs::ops::softmax($a, None, None)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::softmax_device($a, None, None, $stream)
    };

    ($a:expr, $axes:expr) => {
        $crate::mlx_rs::ops::softmax($a, $axes, None)
    };
    ($a:expr, $axes:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::softmax_device($a, $axes, None, $stream)
    };

    ($a:expr, $axes:expr, $precise:expr) => {
        $crate::mlx_rs::ops::softmax($a, $axes, $precise)
    };
    ($a:expr, $axes:expr, $precise:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::softmax_device($a, $axes, $precise, $stream)
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
macro_rules! square {
    ($a:expr) => {
        $a.square()
    };
    ($a:expr, stream=$stream:expr) => {
        $a.square_device($stream)
    };
}

#[macro_export]
macro_rules! subtract {
    ($a:expr, $b:expr) => {
        $a.subtract_device($b.as_ref(), $crate::mlx_rs::StreamOrDevice::default())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.subtract_device($b.as_ref(), $stream)
    };
}

#[macro_export]
macro_rules! tan {
    ($a:expr) => {
        $crate::mlx_rs::ops::tan($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::tan_device($a, $stream)
    };
}

#[macro_export]
macro_rules! tanh {
    ($a:expr) => {
        $crate::mlx_rs::ops::tanh($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::tanh_device($a, $stream)
    };
}

#[macro_export]
macro_rules! block_masked_mm {
    ($a:expr, $b:expr) => {
        $crate::mlx_rs::ops::block_masked_mm($a, $b, None, None, None, None)
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::block_masked_mm_device($a, $b, None, None, None, None, $stream)
    };

    ($a:expr, $b:expr, $block_size:expr) => {
        $crate::mlx_rs::ops::block_masked_mm($a, $b, $block_size, None, None, None)
    };
    ($a:expr, $b:expr, $block_size:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::block_masked_mm_device($a, $b, $block_size, None, None, None, $stream)
    };

    ($a:expr, $b:expr, $block_size:expr, $mask_out:expr) => {
        $crate::mlx_rs::ops::block_masked_mm($a, $b, $block_size, $mask_out, None, None)
    };
    ($a:expr, $b:expr, $block_size:expr, $mask_out:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::block_masked_mm_device($a, $b, $block_size, $mask_out, None, None, $stream)
    };

    ($a:expr, $b:expr, $block_size:expr, $mask_out:expr, $mask_lhs:expr) => {
        $crate::mlx_rs::ops::block_masked_mm($a, $b, $block_size, $mask_out, $mask_lhs, None)
    };
    ($a:expr, $b:expr, $block_size:expr, $mask_out:expr, $mask_lhs:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::block_masked_mm_device($a, $b, $block_size, $mask_out, $mask_lhs, None, $stream)
    };

    ($a:expr, $b:expr, $block_size:expr, $mask_out:expr, $mask_lhs:expr, $mask_rhs:expr) => {
        $crate::mlx_rs::ops::block_masked_mm($a, $b, $block_size, $mask_out, $mask_lhs, $mask_rhs)
    };
    ($a:expr, $b:expr, $block_size:expr, $mask_out:expr, $mask_lhs:expr, $mask_rhs:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::block_masked_mm_device($a, $b, $block_size, $mask_out, $mask_lhs, $mask_rhs, $stream)
    };
}

#[macro_export]
macro_rules! addmm {
    ($a:expr, $b:expr, $c:expr) => {
        $crate::mlx_rs::ops::addmm($a, $b, $c, None, None)
    };
    ($a:expr, $b:expr, $c:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::addmm_device($a, $b, $c, None, None, $stream)
    };

    ($a:expr, $b:expr, $c:expr, $alpha:expr) => {
        $crate::mlx_rs::ops::addmm($a, $b, $c, $alpha, None)
    };
    ($a:expr, $b:expr, $c:expr, $alpha:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::addmm_device($a, $b, $c, $alpha, None, $stream)
    };

    ($a:expr, $b:expr, $c:expr, $alpha:expr, $beta:expr) => {
        $crate::mlx_rs::ops::addmm($a, $b, $c, $alpha, $beta)
    };
    ($a:expr, $b:expr, $c:expr, $alpha:expr, $beta:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::addmm_device($a, $b, $c, $alpha, $beta, $stream)
    };
}

#[macro_export]
macro_rules! inner {
    ($a:expr, $b:expr) => {
        $crate::mlx_rs::ops::inner($a, $b)
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::inner_device($a, $b, $stream)
    };
}

#[macro_export]
macro_rules! outer {
    ($a:expr, $b:expr) => {
        $crate::mlx_rs::ops::outer($a, $b)
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::outer_device($a, $b, $stream)
    };
}

#[macro_export]
macro_rules! tensordot {
    ($a:expr, $b:expr, $axes:expr) => {
        $crate::mlx_rs::ops::tensordot($a, $b, $axes)
    };
    ($a:expr, $b:expr, $axes:expr, stream=$stream:expr) => {
        $crate::mlx_rs::ops::tensordot_device($a, $b, $axes, $stream)
    };
}

#[cfg(test)]
mod tests {
    use mlx_rs::{Array, StreamOrDevice};

    #[test]
    fn test_abs() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[-1, -2, -3], &[3]);

        // We are just testing that the macro compiles
        let _ = abs!(a);
        let _ = abs!(a, stream = stream);
    }

    #[test]
    fn test_acos() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[0.5, 0.6, 0.7], &[3]);

        // We are just testing that the macro compiles
        let _ = acos!(&a);
        let _ = acos!(&a, stream = stream);
    }

    #[test]
    fn test_acosh() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.5, 1.6, 1.7], &[3]);

        // We are just testing that the macro compiles
        let _ = acosh!(&a);
        let _ = acosh!(&a, stream = stream);
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
    fn test_asin() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[0.5, 0.6, 0.7], &[3]);

        // We are just testing that the macro compiles
        let _ = asin!(&a);
        let _ = asin!(&a, stream = stream);
    }

    #[test]
    fn test_asinh() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.5, 1.6, 1.7], &[3]);

        // We are just testing that the macro compiles
        let _ = asinh!(&a);
        let _ = asinh!(&a, stream = stream);
    }

    #[test]
    fn test_atan() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[0.5, 0.6, 0.7], &[3]);

        // We are just testing that the macro compiles
        let _ = atan!(&a);
        let _ = atan!(&a, stream = stream);
    }

    #[test]
    fn test_atanh() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[0.5, 0.6, 0.7], &[3]);

        // We are just testing that the macro compiles
        let _ = atanh!(&a);
        let _ = atanh!(&a, stream = stream);
    }

    #[test]
    fn test_ceil() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.1, 2.2, 3.3], &[3]);

        // We are just testing that the macro compiles
        let _ = ceil!(&a);
        let _ = ceil!(&a, stream = stream);
    }

    #[test]
    fn test_clip() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);

        // We are just testing that the macro compiles
        let _ = clip!(&a, (2, ()));
        let _ = clip!(&a, (2, ()), stream = stream);
    }

    #[test]
    fn test_sub() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        // We are just testing that the macro compiles
        let _ = subtract!(a, b);
        let _ = subtract!(a, b, stream = stream);
    }

    #[test]
    fn test_tan() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[0.5, 0.6, 0.7], &[3]);

        // We are just testing that the macro compiles
        let _ = tan!(&a);
        let _ = tan!(&a, stream = stream);
    }

    #[test]
    fn test_tanh() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[0.5, 0.6, 0.7], &[3]);

        // We are just testing that the macro compiles
        let _ = tanh!(&a);
        let _ = tanh!(&a, stream = stream);
    }

    #[test]
    fn test_neg() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = negative!(a);
        let _ = negative!(a, stream = stream);
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
        let _ = multiply!(a, b);
        let _ = multiply!(a, b, stream = stream);
    }

    #[test]
    fn test_div() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        // We are just testing that the macro compiles
        let _ = divide!(a, b);
        let _ = divide!(a, b, stream = stream);
    }

    #[test]
    fn test_pow() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        // We are just testing that the macro compiles
        let _ = power!(a, b);
        let _ = power!(a, b, stream = stream);
    }

    #[test]
    fn test_rem() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        // We are just testing that the macro compiles
        let _ = remainder!(a, b);
        let _ = remainder!(a, b, stream = stream);
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
    fn test_cosh() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = cosh!(&a);
        let _ = cosh!(&a, stream = stream);
    }

    #[test]
    fn test_erf() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[0.5, 0.6, 0.7], &[3]);

        // We are just testing that the macro compiles
        let _ = erf!(&a);
        let _ = erf!(&a, stream = stream);
    }

    #[test]
    fn test_erfinv() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[0.5, 0.6, 0.7], &[3]);

        // We are just testing that the macro compiles
        let _ = erfinv!(&a);
        let _ = erfinv!(&a, stream = stream);
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
    fn test_expm1() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = expm1!(&a);
        let _ = expm1!(&a, stream = stream);
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
    fn test_floor_divide() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[4, 5, 6], &[3]);

        // We are just testing that the macro compiles
        let _ = floor_divde!(&a, &b);
        let _ = floor_divde!(&a, &b, stream = stream);
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
    fn test_log_add_exp() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        // We are just testing that the macro compiles
        let _ = log_add_exp!(&a, &b);
        let _ = log_add_exp!(&a, &b, stream = stream);
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
    fn test_maximum() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[4, 5, 6], &[3]);

        // We are just testing that the macro compiles
        let _ = maximum!(&a, &b);
        let _ = maximum!(&a, &b, stream = stream);
    }

    #[test]
    fn test_minimum() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[4, 5, 6], &[3]);

        // We are just testing that the macro compiles
        let _ = minimum!(&a, &b);
        let _ = minimum!(&a, &b, stream = stream);
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
    fn test_sigmoid() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);

        // We are just testing that the macro compiles
        let _ = sigmoid!(&a);
        let _ = sigmoid!(&a, stream = stream);
    }

    #[test]
    fn test_sign() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, -2, 3], &[3]);

        // We are just testing that the macro compiles
        let _ = sign!(&a);
        let _ = sign!(&a, stream = stream);
    }

    #[test]
    fn test_softmax() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);

        // We are just testing that the macro compiles
        let _ = softmax!(&a);
        let _ = softmax!(&a, &[0]);
        let _ = softmax!(&a, &[0], true);
        let _ = softmax!(&a, stream = &stream);
        let _ = softmax!(&a, &[0], stream = &stream);
        let _ = softmax!(&a, &[0], true, stream = &stream);
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
    fn test_sinh() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macro compiles
        let _ = sinh!(&a);
        let _ = sinh!(&a, stream = stream);
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
