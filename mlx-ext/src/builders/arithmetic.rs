use mlx_rs::{
    error::Exception,
    ops::{ClipBound, TensorDotDims},
    Array, StreamOrDevice,
};

/* -------------------------------------------------------------------------- */
/*          Unary ops that takes exactly one input array as argument          */
/* -------------------------------------------------------------------------- */

macro_rules! unary_op_builder {
    ($name:ident, $op:ident, $ret:ty) => {
        pub struct $name {
            pub stream: Option<StreamOrDevice>,
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $name {
            pub fn new() -> Self {
                Self { stream: None }
            }

            pub fn stream(mut self, stream: StreamOrDevice) -> Self {
                self.stream = Some(stream);
                self
            }

            pub fn build(self, a: impl AsRef<Array>) -> $ret {
                match self.stream {
                    Some(stream) => paste::paste! {
                        mlx_rs::ops::[<$op _device>](a.as_ref(), stream)
                    },
                    None => mlx_rs::ops::$op(a.as_ref()),
                }
            }
        }
    };
}

unary_op_builder!(Abs, abs, Array);
unary_op_builder!(Acos, acos, Array);
unary_op_builder!(Acosh, acosh, Array);
unary_op_builder!(Asin, asin, Array);
unary_op_builder!(Asinh, asinh, Array);
unary_op_builder!(Atan, atan, Array);
unary_op_builder!(Atanh, atanh, Array);
unary_op_builder!(Ceil, ceil, Result<Array, Exception>);
unary_op_builder!(Cos, cos, Array);
unary_op_builder!(Cosh, cosh, Array);
unary_op_builder!(Degrees, degrees, Array);
unary_op_builder!(Erf, erf, Array);
unary_op_builder!(Erfinv, erfinv, Array);
unary_op_builder!(Exp, exp, Array);
unary_op_builder!(Expm1, expm1, Array);
unary_op_builder!(Floor, floor, Result<Array, Exception>);
unary_op_builder!(Log, log, Array);
unary_op_builder!(Log10, log10, Array);
unary_op_builder!(Log1p, log1p, Array);
unary_op_builder!(Log2, log2, Array);
unary_op_builder!(Negative, negative, Result<Array, Exception>);
unary_op_builder!(Radians, radians, Array);
unary_op_builder!(Reciprocal, reciprocal, Array);
unary_op_builder!(Rsqrt, rsqrt, Array);
unary_op_builder!(Sigmoid, sigmoid, Array);
unary_op_builder!(Sign, sign, Array);
unary_op_builder!(Sin, sin, Array);
unary_op_builder!(Sinh, sinh, Array);
unary_op_builder!(Sqrt, sqrt, Array);
unary_op_builder!(Square, square, Array);
unary_op_builder!(Tan, tan, Array);
unary_op_builder!(Tanh, tanh, Array);

/* -------------------------------------------------------------------------- */
/*        Unary ops that take one input array and some other arguments        */
/* -------------------------------------------------------------------------- */

pub struct Clip {
    pub stream: Option<StreamOrDevice>,
}

impl Default for Clip {
    fn default() -> Self {
        Self::new()
    }
}

impl Clip {
    pub fn new() -> Self {
        Self { stream: None }
    }

    pub fn stream(mut self, stream: StreamOrDevice) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn build<'min, 'max>(
        self,
        a: impl AsRef<Array>,
        bound: impl ClipBound<'min, 'max>,
    ) -> Result<Array, Exception> {
        match self.stream {
            Some(stream) => mlx_rs::ops::clip_device(a.as_ref(), bound, stream),
            None => mlx_rs::ops::clip(a.as_ref(), bound),
        }
    }
}

pub struct Round {
    pub stream: Option<StreamOrDevice>,
    pub decimals: Option<i32>,
}

impl Default for Round {
    fn default() -> Self {
        Self::new()
    }
}

impl Round {
    pub fn new() -> Self {
        Self {
            stream: None,
            decimals: None,
        }
    }

    pub fn stream(mut self, stream: StreamOrDevice) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn decimals(mut self, decimals: i32) -> Self {
        self.decimals = Some(decimals);
        self
    }

    pub fn build(self, a: impl AsRef<Array>) -> Array {
        match self.stream {
            Some(stream) => mlx_rs::ops::round_device(a.as_ref(), self.decimals, stream),
            None => mlx_rs::ops::round(a.as_ref(), self.decimals),
        }
    }
}

pub struct Softmax<'a> {
    pub stream: Option<StreamOrDevice>,
    pub axes: Option<&'a [i32]>,
    pub precise: Option<bool>,
}

impl<'a> Default for Softmax<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Softmax<'a> {
    pub fn new() -> Self {
        Self {
            stream: None,
            axes: None,
            precise: None,
        }
    }

    pub fn stream(mut self, stream: StreamOrDevice) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn axes(mut self, axes: &'a [i32]) -> Self {
        self.axes = Some(axes);
        self
    }

    pub fn precise(mut self, precise: bool) -> Self {
        self.precise = Some(precise);
        self
    }

    pub fn build(self, a: impl AsRef<Array>) -> Array {
        match self.stream {
            Some(stream) => {
                mlx_rs::ops::softmax_device(a.as_ref(), self.axes, self.precise, stream)
            }
            None => mlx_rs::ops::softmax(a.as_ref(), self.axes, self.precise),
        }
    }
}

/* -------------------------------------------------------------------------- */
/*         Binary ops that take exactly two input arrays as arguments         */
/* -------------------------------------------------------------------------- */

macro_rules! binary_op_builder {
    ($name:ident, $op:ident, $ret:ty) => {
        pub struct $name {
            pub stream: Option<StreamOrDevice>,
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $name {
            pub fn new() -> Self {
                Self { stream: None }
            }

            pub fn stream(mut self, stream: StreamOrDevice) -> Self {
                self.stream = Some(stream);
                self
            }

            pub fn build(self, a: impl AsRef<Array>, b: impl AsRef<Array>) -> $ret {
                match self.stream {
                    Some(stream) => paste::paste! {
                        mlx_rs::ops::[<$op _device>](a.as_ref(), b.as_ref(), stream)
                    },
                    None => mlx_rs::ops::$op(a.as_ref(), b.as_ref()),
                }
            }
        }
    };
}

binary_op_builder!(Add, add, Result<Array, Exception>);
binary_op_builder!(Divide, divide, Result<Array, Exception>);
binary_op_builder!(Divmod, divmod, Result<(Array, Array), Exception>);
binary_op_builder!(LogAddExp, log_add_exp, Result<Array, Exception>);
binary_op_builder!(Matmul, matmul, Result<Array, Exception>);
binary_op_builder!(Maximum, maximum, Result<Array, Exception>);
binary_op_builder!(Minimum, minimum, Result<Array, Exception>);
binary_op_builder!(Multiply, multiply, Result<Array, Exception>);
binary_op_builder!(Power, power, Result<Array, Exception>);
binary_op_builder!(Remainder, remainder, Result<Array, Exception>);
binary_op_builder!(Subtract, subtract, Result<Array, Exception>);
binary_op_builder!(Inner, inner, Result<Array, Exception>);
binary_op_builder!(Outer, outer, Result<Array, Exception>);

/* -------------------------------------------------------------------------- */
/*          Binary ops that take two input arrays and some other args         */
/* -------------------------------------------------------------------------- */

pub struct BlockMaskedMm<'mo, 'lhs, 'rhs> {
    pub stream: Option<StreamOrDevice>,
    pub block_size: Option<i32>,
    pub mask_out: Option<&'mo Array>,
    pub mask_lfs: Option<&'lhs Array>,
    pub mask_rfs: Option<&'rhs Array>,
}

impl<'mo, 'lhs, 'rhs> Default for BlockMaskedMm<'mo, 'lhs, 'rhs> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'mo, 'lhs, 'rhs> BlockMaskedMm<'mo, 'lhs, 'rhs> {
    pub fn new() -> Self {
        Self {
            stream: None,
            block_size: None,
            mask_out: None,
            mask_lfs: None,
            mask_rfs: None,
        }
    }

    pub fn stream(mut self, stream: StreamOrDevice) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn block_size(mut self, block_size: i32) -> Self {
        self.block_size = Some(block_size);
        self
    }

    pub fn mask_out(mut self, mask_out: &'mo Array) -> Self {
        self.mask_out = Some(mask_out);
        self
    }

    pub fn mask_lfs(mut self, mask_lfs: &'lhs Array) -> Self {
        self.mask_lfs = Some(mask_lfs);
        self
    }

    pub fn mask_rfs(mut self, mask_rfs: &'rhs Array) -> Self {
        self.mask_rfs = Some(mask_rfs);
        self
    }

    pub fn build(self, a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array, Exception> {
        match self.stream {
            Some(stream) => mlx_rs::ops::block_masked_mm_device(
                a.as_ref(),
                b.as_ref(),
                self.block_size,
                self.mask_out,
                self.mask_lfs,
                self.mask_rfs,
                stream,
            ),
            None => mlx_rs::ops::block_masked_mm(
                a.as_ref(),
                b.as_ref(),
                self.block_size,
                self.mask_out,
                self.mask_lfs,
                self.mask_rfs,
            ),
        }
    }
}

pub struct Tensordot {
    pub stream: Option<StreamOrDevice>,
}

impl Default for Tensordot {
    fn default() -> Self {
        Self::new()
    }
}

impl Tensordot {
    pub fn new() -> Self {
        Self { stream: None }
    }

    pub fn stream(mut self, stream: StreamOrDevice) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn build<'a>(
        self,
        a: impl AsRef<Array>,
        b: impl AsRef<Array>,
        axes: impl Into<TensorDotDims<'a>>,
    ) -> Result<Array, Exception> {
        match self.stream {
            Some(stream) => mlx_rs::ops::tensordot_device(a.as_ref(), b.as_ref(), axes, stream),
            None => mlx_rs::ops::tensordot(a.as_ref(), b.as_ref(), axes),
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                                 Ternary ops                                */
/* -------------------------------------------------------------------------- */

pub struct Addmm {
    pub stream: Option<StreamOrDevice>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,
}

impl Default for Addmm {
    fn default() -> Self {
        Self::new()
    }
}

impl Addmm {
    pub fn new() -> Self {
        Self {
            stream: None,
            alpha: None,
            beta: None,
        }
    }

    pub fn stream(mut self, stream: StreamOrDevice) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = Some(alpha);
        self
    }

    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = Some(beta);
        self
    }

    pub fn build(
        self,
        a: impl AsRef<Array>,
        b: impl AsRef<Array>,
        c: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        match self.stream {
            Some(stream) => mlx_rs::ops::addmm_device(
                a.as_ref(),
                b.as_ref(),
                c.as_ref(),
                self.alpha,
                self.beta,
                stream,
            ),
            None => mlx_rs::ops::addmm(a.as_ref(), b.as_ref(), c.as_ref(), self.alpha, self.beta),
        }
    }
}

#[cfg(test)]
mod tests {
    use mlx_rs::{array, StreamOrDevice};

    use super::*;

    #[test]
    fn test_abs() {
        let stream = StreamOrDevice::default();
        let a = array!([-1, -2, -3]);

        let _output = Abs::new().build(&a);
        let _output = Abs::new().stream(stream).build(&a);
    }

    #[test]
    fn test_clip() {
        let stream = StreamOrDevice::default();
        let a = array!([1, 2, 3]);

        let _output = Clip::new().build(&a, (0, 2));
        let _output = Clip::new().stream(stream).build(&a, (0, 2));
    }

    #[test]
    fn test_softmax() {
        let stream = StreamOrDevice::default();
        let a = array!([[1, 2], [3, 4]]);

        let _output = Softmax::new().build(&a);
        let _output = Softmax::new().stream(stream).build(&a);
        let _output = Softmax::new().axes(&[0]).build(&a);
        let _output = Softmax::new().precise(true).build(&a);
        let _output = Softmax::new().axes(&[0]).precise(true).build(&a);
    }
}
