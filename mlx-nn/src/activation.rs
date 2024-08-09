use std::f32::consts::PI;

use mlx_rs::{array, error::Exception, ops::log_sum_exp, transforms::compile::compile, Array};

/// Applies the element-wise sigmoid logistic sigmoid.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sigmoid.html)
///
/// This is:
///
/// ```rust
/// mlx_rs::ops::sigmoid(x)
/// ```
pub fn sigmoid(x: impl AsRef<Array>) -> Array {
    mlx_rs::ops::sigmoid(x.as_ref())
}

/// Applies the Rectified Linear Unit.
///
/// This is:
///
/// ```rust
/// mlx_rs::ops::maximum(x, 0)
/// ```
pub fn relu(x: impl AsRef<Array>) -> Result<Array, Exception> {
    mlx_rs::ops::maximum(x.as_ref(), 0)
}

/// Applies the Leaky Rectified Linear Unit.
///
/// `neg_slope` is default to 0.01 if not provided.
///
/// This is:
///
/// ```rust
/// mlx_rs::ops::maximum(neg_slope * x, x)
/// ```
pub fn leaky_relu(
    x: impl AsRef<Array>,
    neg_slope: impl Into<Option<f32>>,
) -> Result<Array, Exception> {
    let neg_slope = array!(neg_slope.into().unwrap_or(0.01));
    // We have to use this indirection, otherwise the compiler cannot
    // infer the lifetime of the value returned by the closure properly
    compiled_leaky_relu(x.as_ref(), &neg_slope)
}

/// Applies the Log Softmax function.
///
/// This is:
///
/// ```rust
/// x - log_sum_exp(x, axis, true)
/// ```
pub fn log_softmax(x: impl AsRef<Array>, axis: impl Into<Option<i32>>) -> Result<Array, Exception> {
    let x = x.as_ref();
    let axis = axis.into().unwrap_or(-1);
    x.subtract(log_sum_exp(x, &[axis], true)?)
}

pub fn elu(x: impl AsRef<Array>, alpha: impl Into<Option<f32>>) -> Result<Array, Exception> {
    let alpha = array!(alpha.into().unwrap_or(1.0));
    // We have to use this indirection, otherwise the compiler cannot
    // infer the lifetime of the value returned by the closure properly
    compiled_elu(x.as_ref(), &alpha)
}

pub fn glu(x: impl AsRef<Array>, axis: impl Into<Option<i32>>) -> Result<Array, Exception> {
    let split = x.as_ref().split_equal(2, axis)?;
    let (a, b) = (&split[0], &split[1]);
    Ok(a * sigmoid(b))
}

/// Applies the Exponential Linear Unit.
///
/// This is:
///
/// ```rust
/// mlx_rs::ops::log_add_exp(x, 0)
/// ```
pub fn softplus(x: impl AsRef<Array>) -> Result<Array, Exception> {
    mlx_rs::ops::log_add_exp(x.as_ref(), 0)
}

/* -------------------------------------------------------------------------- */
/*                        Compiled activation functions                       */
/* -------------------------------------------------------------------------- */

#[inline]
fn compiled_leaky_relu(x: &Array, neg_slope: &Array) -> Result<Array, Exception> {
    let f = |(x_, neg_slope_): (&Array, &Array)| {
        // This will not panic because a scalar can always be broadcasted to any shape
        mlx_rs::ops::maximum(neg_slope_ * x_, x_).unwrap()
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled((x, neg_slope))
}

#[inline]
fn compiled_elu(x: &Array, alpha: &Array) -> Result<Array, Exception> {
    let f = |(x_, alpha_): (&Array, &Array)| {
        mlx_rs::ops::which(&x_.gt(0).unwrap(), x_, alpha_ * (mlx_rs::ops::exp(x_) - 1)).unwrap()
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled((x, alpha))
}

#[inline]
fn compiled_relu6(x: &Array) -> Result<Array, Exception> {
    let f = |x_: &Array| mlx_rs::ops::minimum(&mlx_rs::ops::maximum(x_, 0).unwrap(), 6).unwrap();
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_softsign(x: &Array) -> Result<Array, Exception> {
    let f = |x_: &Array| x_ / (mlx_rs::ops::abs(x_).add(1).unwrap());
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_celu(x: &Array, alpha: &Array) -> Result<Array, Exception> {
    let f = |(x_, alpha_): (&Array, &Array)| {
        mlx_rs::ops::maximum(x_, 0.0).unwrap()
            + alpha_ * (mlx_rs::ops::exp(&(mlx_rs::ops::minimum(x_, 0.0).unwrap() / alpha_)) - 1)
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled((x, alpha))
}

#[inline]
fn compiled_silu(x: &Array) -> Result<Array, Exception> {
    let f = |x_: &Array| x_ * sigmoid(x_);
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_log_sigmoid(x: &Array) -> Result<Array, Exception> {
    let f = |x_: &Array| -softplus(&(-x_)).unwrap();
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_gelu(x: &Array) -> Result<Array, Exception> {
    use mlx_rs::ops::erf;
    let f = |x_: &Array| x_ * (erf(&(x_ / 2f32.sqrt())) + 1) / 2;
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_gelu_approximate(x: &Array) -> Result<Array, Exception> {
    use mlx_rs::ops::{sqrt, tanh};

    let f = move |x_: &Array| {
        // 0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x ** 3)))
        array!(0.5)
            * x_
            * (array!(0.5)
                + tanh(&(sqrt(&array!(2.0 / PI)) * (x_ + array!(0.044715) * x_.power(3).unwrap()))))
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_gelu_fast_approximate(x: &Array) -> Result<Array, Exception> {
    let f = |x_: &Array| x_ * sigmoid(&(array!(1.773) * x_));
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_selu(x: &Array) -> Result<Array, Exception> {
    let f = |x_: &Array| elu(x_, 1.67326).unwrap() * 1.0507;
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_prelu(x: &Array, alpha: &Array) -> Result<Array, Exception> {
    let f = |(x_, alpha_): (&Array, &Array)| {
        mlx_rs::ops::maximum(0, x_).unwrap() + alpha_ * mlx_rs::ops::minimum(0, x_).unwrap()
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled((x, alpha))
}

#[inline]
fn compiled_mish(x: &Array) -> Result<Array, Exception> {
    use mlx_rs::ops::tanh;

    let f = |x_: &Array| x_ * tanh(&softplus(x_).unwrap());
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_hard_swish(x: &Array) -> Result<Array, Exception> {
    let f = |x_: &Array| {
        let max_x_plus_3 = mlx_rs::ops::maximum(&(x_ + 3), 0).unwrap();
        x_ * mlx_rs::ops::minimum(&max_x_plus_3, 6).unwrap() / 6
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}
