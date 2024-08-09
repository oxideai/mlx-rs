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

/// Applies the Exponential Linear Unit.
/// 
/// This is:
/// 
/// ```rust
/// which(x.gt(0), x, alpha * (exp(x) - 1))
/// ```
/// 
/// # Arguments
/// 
/// - `x`: The input array
/// - `alpha`: Default to 1.0 if not provided
pub fn elu(x: impl AsRef<Array>, alpha: impl Into<Option<f32>>) -> Result<Array, Exception> {
    let alpha = array!(alpha.into().unwrap_or(1.0));
    // We have to use this indirection, otherwise the compiler cannot
    // infer the lifetime of the value returned by the closure properly
    compiled_elu(x.as_ref(), &alpha)
}

/// Applies the Rectified Linear Unit 6.
/// 
/// This is:
/// 
/// ```rust
/// minimum(maximum(x, 0), 6)
/// ```
pub fn relu6(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_relu6(x.as_ref())
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

/// Applies the Softsign function.
/// 
/// This is:
/// 
/// ```rust
/// x / (1 + abs(x))
/// ```
pub fn softsign(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_softsign(x.as_ref())
}

/// Applies the Continuously Differentiable Exponential Linear Unit.
/// 
/// This is:
/// 
/// ```rust
/// maximum(x, 0) + alpha * (exp(minimum(x, 0) / alpha) - 1)
/// ```
pub fn celu(x: impl AsRef<Array>, alpha: impl Into<Option<f32>>) -> Result<Array, Exception> {
    let alpha = array!(alpha.into().unwrap_or(1.0));
    // We have to use this indirection, otherwise the compiler cannot
    // infer the lifetime of the value returned by the closure properly
    compiled_celu(x.as_ref(), &alpha)
}

/// Applies the Sigmoid Linear Unit. Also known as Swish.
/// 
/// This is:
/// 
/// ```rust
/// x * sigmoid(x)
/// ```
pub fn silu(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_silu(x.as_ref())
}

/// Applies the Log Sigmoid function.
/// 
/// This is:
/// 
/// ```rust
/// -softplus(-x)
/// ```
pub fn log_sigmoid(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_log_sigmoid(x.as_ref())
}

/// Applies the Gaussian Error Linear Units function.
/// 
/// This is:
/// 
/// ```rust
/// x * (1 + erf(x / 2.sqrt())) / 2
/// ```
pub fn gelu(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_gelu(x.as_ref())
}

/// An approximation to Gaussian Error Linear Unit.
/// 
/// This is:
/// 
/// ```rust
/// 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x ** 3)))
/// ```
pub fn gelu_approximate(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_gelu_approximate(x.as_ref())
}

/// A fast approximation to Gaussian Error Linear Unit.
/// 
/// This is:
/// 
/// ```rust
/// x * sigmoid(1.773 * x)
/// ```
pub fn gelu_fast_approximate(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_gelu_fast_approximate(x.as_ref())
}

/// Applies the gated linear unit function.
/// 
/// This function splits the `axis` dimension of the input into two halves
/// (`a` and `b`) and applies `a * sigmoid(b)`.
pub fn glu(x: impl AsRef<Array>, axis: impl Into<Option<i32>>) -> Result<Array, Exception> {
    let split = x.as_ref().split_equal(2, axis)?;
    let (a, b) = (&split[0], &split[1]);
    Ok(a * sigmoid(b))
}

/// Applies the Step Activation Function.
///
/// This function implements a binary step activation, where the output is set
/// to 1 if the input is greater than a specified threshold, and 0 otherwise.
///
/// This is:
///
/// ```rust
/// where(x.gt(threshold), 1, 0)
/// ```
pub fn step(x: impl AsRef<Array>, threshold: impl Into<Option<f32>>) -> Result<Array, Exception> {
    let threshold = threshold.into().unwrap_or(0.0);
    mlx_rs::ops::r#where(&x.as_ref().gt(threshold)?, 1, 0)
}

/// Applies the Scaled Exponential Linear Unit.
/// 
/// This is:
/// 
/// ```rust
/// elu(x, 1.67326) * 1.0507
/// ```
pub fn selu(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_selu(x.as_ref())
}

/// Applies the element-wise parametric ReLU.
/// 
/// This is:
/// 
/// ```rust
/// maximum(0, x) + alpha * minimum(0, x)
/// ```
pub fn prelu(x: impl AsRef<Array>, alpha: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_prelu(x.as_ref(), alpha.as_ref())
}

/// Applies the Mish function, element-wise.
/// 
/// Mish: A Self Regularized Non-Monotonic Neural Activation Function.
///
/// Reference: [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)
///
/// This is:
///
/// ```rust
/// x * tanh(softplus(x))
/// ```
pub fn mish(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_mish(x.as_ref())
}

/// Applies the hardswish function, element-wise.
/// 
/// This is:
/// 
/// ```rust
/// x * minimum(maximum(x + 3, 0), 6) / 6
/// ```
pub fn hard_swish(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_hard_swish(x.as_ref())
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
    let f = |x_: &Array| x_ / (array!(1) + mlx_rs::ops::abs(x_));
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
    let f = |x_: &Array| x_ * (array!(1) + erf(&(x_ / 2f32.sqrt()))) / 2;
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
            * (array!(1.0)
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
