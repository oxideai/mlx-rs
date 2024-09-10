use std::f32::consts::PI;

use mlx_macros::ModuleParameters;
use mlx_nn_module::{Module, Param};
use mlx_rs::{array, error::Exception, ops::log_sum_exp, transforms::compile::compile, Array};

/// Applies the element-wise sigmoid logistic sigmoid.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sigmoid.html)
///
/// This is:
///
/// ```rust, ignore
/// sigmoid(x)
/// ```
pub fn sigmoid(x: impl AsRef<Array>) -> Array {
    mlx_rs::ops::sigmoid(x.as_ref())
}

/// Applies the Rectified Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// maximum(x, 0)
/// ```
pub fn relu(x: impl AsRef<Array>) -> Result<Array, Exception> {
    mlx_rs::ops::maximum(x.as_ref(), &array!(0))
}

/// Applies the Leaky Rectified Linear Unit.
///
/// `neg_slope` is default to 0.01 if not provided.
///
/// This is:
///
/// ```rust, ignore
/// maximum(neg_slope * x, x)
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
/// ```rust, ignore
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
/// ```rust, ignore
/// which(x.gt(0), x, alpha * (exp(x) - 1))
/// ```
///
/// # Params
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
/// ```rust, ignore
/// minimum(maximum(x, 0), 6)
/// ```
pub fn relu6(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_relu6(x.as_ref())
}

/// Applies the Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// log_add_exp(x, 0)
/// ```
pub fn softplus(x: impl AsRef<Array>) -> Result<Array, Exception> {
    mlx_rs::ops::log_add_exp(x.as_ref(), array!(0))
}

/// Applies the Softsign function.
///
/// This is:
///
/// ```rust, ignore
/// x / (1 + abs(x))
/// ```
pub fn softsign(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_softsign(x.as_ref())
}

/// Applies the Continuously Differentiable Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
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
/// ```rust, ignore
/// x * sigmoid(x)
/// ```
pub fn silu(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_silu(x.as_ref())
}

/// Applies the Log Sigmoid function.
///
/// This is:
///
/// ```rust, ignore
/// -softplus(-x)
/// ```
pub fn log_sigmoid(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_log_sigmoid(x.as_ref())
}

/// Applies the Gaussian Error Linear Units function.
///
/// This is:
///
/// ```rust, ignore
/// x * (1 + erf(x / 2.sqrt())) / 2
/// ```
pub fn gelu(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_gelu(x.as_ref())
}

/// An approximation to Gaussian Error Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x ** 3)))
/// ```
pub fn gelu_approximate(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_gelu_approximate(x.as_ref())
}

/// A fast approximation to Gaussian Error Linear Unit.
///
/// This is:
///
/// ```rust, ignore
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
/// ```rust, ignore
/// r#where(x.gt(threshold), 1, 0)
/// ```
pub fn step(x: impl AsRef<Array>, threshold: impl Into<Option<f32>>) -> Result<Array, Exception> {
    let threshold = threshold.into().unwrap_or(0.0);
    mlx_rs::ops::r#where(&x.as_ref().gt(threshold)?, array!(1), array!(0))
}

/// Applies the Scaled Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// elu(x, 1.67326) * 1.0507
/// ```
pub fn selu(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_selu(x.as_ref())
}

/// Applies the element-wise parametric ReLU.
///
/// This is:
///
/// ```rust, ignore
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
/// ```rust, ignore
/// x * tanh(softplus(x))
/// ```
pub fn mish(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_mish(x.as_ref())
}

/// Applies the hardswish function, element-wise.
///
/// This is:
///
/// ```rust, ignore
/// x * minimum(maximum(x + 3, 0), 6) / 6
/// ```
pub fn hard_swish(x: impl AsRef<Array>) -> Result<Array, Exception> {
    compiled_hard_swish(x.as_ref())
}

/// Applies the gated linear unit function.
///
/// This splits the `axis` dimension of the input into two halves
/// (`a` and `b`) and applies `a * sigmoid(b)`.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Glu {
    /// The axis to split the input tensor. Default to -1 if not provided.
    pub axis: Option<i32>,
}

impl Default for Glu {
    fn default() -> Self {
        Self::new()
    }
}

impl Glu {
    /// Creates a [`Glu`] module.
    ///
    /// # Params
    ///
    /// - `axis`: The axis to split the input tensor. Default to -1 if not provided.
    pub fn new() -> Self {
        Self { axis: None }
    }

    /// Sets the value of the `axis` field.
    pub fn with_axis(mut self, axis: impl Into<Option<i32>>) -> Self {
        self.axis = axis.into();
        self
    }
}

impl Module for Glu {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        glu(x, self.axis)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the element-wise logistic sigmoid.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sigmoid.html)
///
/// This is:
///
/// ```rust, ignore
/// sigmoid(x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Sigmoid;

// We implement this just for the sake of consistency
impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Sigmoid {
    /// Creates a new [`Sigmoid`] module.
    /// 
    /// This is just to be consistent with the other modules.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Sigmoid {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        Ok(sigmoid(x))
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Mish function, element-wise.
///
/// Mish: A Self Regularized Non-Monotonic Neural Activation Function.
///
/// Reference: [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)
///
/// This is:
///
/// ```rust, ignore
/// x * tanh(softplus(x))
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Mish;

impl Default for Mish {
    fn default() -> Self {
        Self::new()
    }
}

impl Mish {
    /// Creates a new [`Mish`] module.
    /// 
    /// This is just to be consistent with the other modules.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Mish {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        mish(x)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Rectified Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// maximum(x, 0)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Relu;

impl Default for Relu {
    fn default() -> Self {
        Self::new()
    }
}

impl Relu {
    /// Creates a new [`Relu`] module.
    /// 
    /// This is just to be consistent with the other modules.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Relu {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        relu(x)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Leaky Rectified Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// maximum(neg_slope * x, x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct LeakyReLU {
    /// The negative slope. Default to 0.01 if not provided.
    pub neg_slope: Option<f32>,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl LeakyReLU {
    /// Creates a new [`LeakyReLU`] module.
    pub fn new() -> Self {
        Self { neg_slope: None }
    }

    /// Sets the value of the `neg_slope`
    pub fn with_neg_slope(mut self, neg_slope: impl Into<Option<f32>>) -> Self {
        self.neg_slope = neg_slope.into();
        self
    }
}

impl Module for LeakyReLU {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        leaky_relu(x, self.neg_slope)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Rectified Linear Unit 6.
///
/// This is:
///
/// ```rust, ignore
/// minimum(&maximum(x, 0).unwrap(), 6).unwrap()
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Relu6;

impl Default for Relu6 {
    fn default() -> Self {
        Self::new()
    }
}

impl Relu6 {
    /// Creates a new [`Relu6`] module.
    /// 
    /// This is just to be consistent with the other modules.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Relu6 {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        relu6(x)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Softmax function.
///
/// This is:
///
/// ```rust, ignore
/// softmax(&x, None, None)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Softmax {
    /// The axis to apply the softmax.
    pub axis: Option<i32>,
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Softmax {
    /// Creates a new [`Softmax`] module.
    pub fn new() -> Self {
        Self { axis: None }
    }

    /// Sets the value of the `axis`
    pub fn with_axis(mut self, axis: impl Into<Option<i32>>) -> Self {
        self.axis = axis.into();
        self
    }
}

impl Module for Softmax {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        match self.axis {
            Some(axis) => Ok(mlx_rs::ops::softmax(x, &[axis], None)),
            None => Ok(mlx_rs::ops::softmax(x, None, None)),
        }
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Softplus function.
///
/// This is:
///
/// ```rust, ignore
/// log_add_exp(x, 0)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Softplus;

impl Default for Softplus {
    fn default() -> Self {
        Self::new()
    }
}

impl Softplus {
    /// Creates a new [`Softplus`] module.
    /// 
    /// This is just to be consistent with the other modules.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Softplus {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        softplus(x)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Softsign function.
///
/// This is:
///
/// ```rust, ignore
/// x / (array!(1) + abs(x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Softsign;

impl Default for Softsign {
    fn default() -> Self {
        Self::new()
    }
}

impl Softsign {
    /// Creates a new [`Softsign`] module.
    /// 
    /// This is just to be consistent with the other modules.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Softsign {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        softsign(x)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Continuously Differentiable Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// maximum(x, 0.0).unwrap()
///     + alpha * (exp(&(minimum(x, 0.0).unwrap() / alpha)) - 1)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Celu {
    /// The alpha value. See [`celu`] for more details.
    pub alpha: Option<f32>,
}

impl Default for Celu {
    fn default() -> Self {
        Self::new()
    }
}

impl Celu {
    /// Creates a new [`Celu`] module.
    pub fn new() -> Self {
        Self { alpha: None }
    }

    /// Sets the value of the `alpha`
    pub fn with_alpha(mut self, alpha: impl Into<Option<f32>>) -> Self {
        self.alpha = alpha.into();
        self
    }
}

impl Module for Celu {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        celu(x, self.alpha)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Sigmoid Linear Unit. Also known as Swish.
///
/// This is:
///
/// ```rust, ignore
/// x * sigmoid(x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Silu;

impl Default for Silu {
    fn default() -> Self {
        Self::new()
    }
}

impl Silu {
    /// Creates a new [`Silu`] module.
    /// 
    /// This is just to be consistent with the other modules.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Silu {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        silu(x)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Log Softmax function.
///
/// This is:
///
/// ```rust, ignore
/// x - log_sum_exp(x, axis, true)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct LogSoftmax {
    /// The axis value. See [`log_softmax`] for more details.
    pub axis: Option<i32>,
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSoftmax {
    /// Creates a new [`LogSoftmax`] module.
    pub fn new() -> Self {
        Self { axis: None }
    }

    /// Sets the value of the `axis`
    pub fn with_axis(mut self, axis: impl Into<Option<i32>>) -> Self {
        self.axis = axis.into();
        self
    }
}

impl Module for LogSoftmax {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        log_softmax(x, self.axis)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Log Sigmoid function.
///
/// This is:
///
/// ```rust, ignore
/// -softplus(-x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct LogSigmoid;

impl Default for LogSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSigmoid {
    /// Creates a new [`LogSigmoid`] module.
    /// 
    /// This is just to be consistent with the other modules.
    pub fn new() -> Self {
        Self
    }
}

impl Module for LogSigmoid {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        log_sigmoid(x)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the element-wise parametric ReLU.
///
/// This is:
///
/// ```rust, ignore
/// maximum(0, x) + alpha * minimum(0, x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Prelu {
    /// The alpha value. See [`prelu`] for more details.
    #[param]
    pub alpha: Param<Array>, // TODO: double check if this is trainable
}

impl Prelu {
    /// Creates a new [`Prelu`] module.
    pub fn new(alpha: Array) -> Self {
        Self {
            alpha: Param::new(alpha),
        }
    }
}

impl Module for Prelu {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        prelu(x, &self.alpha)
    }

    fn train(&mut self, _: bool) {}
}

/// Variants of Gaussian Error Linear Units function.
#[derive(Debug, Clone, Copy, Default)]
pub enum GeluApprox {
    /// Uses [`gelu`]
    #[default]
    None,

    /// Uses [`gelu_approximate`]
    Precise,

    /// Uses [`gelu_fast_approximate`]
    Fast,
}

/// Applies the Gaussian Error Linear Units function.
///
/// There are three variants:
///
/// - `GeluApprox::None`: Uses [`gelu`]. This is the default.
/// - `GeluApprox::Precise`: Uses [`gelu_approximate`]
/// - `GeluApprox::Fast`: Uses [`gelu_fast_approximate`]
#[derive(Debug, Clone, ModuleParameters)]
pub struct Gelu {
    /// The approximation to use. Default to `GeluApprox::None` if not provided.
    pub approximate: GeluApprox,
}

impl Default for Gelu {
    fn default() -> Self {
        Self::new()
    }
}

impl Gelu {
    /// Creates a new [`Gelu`] module.
    pub fn new() -> Self {
        Self {
            approximate: GeluApprox::None,
        }
    }

    /// Sets the value of the `approximate`
    pub fn approximate(mut self, approximate: GeluApprox) -> Self {
        self.approximate = approximate;
        self
    }
}

impl Module for Gelu {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        match self.approximate {
            GeluApprox::None => gelu(x),
            GeluApprox::Precise => gelu_approximate(x),
            GeluApprox::Fast => gelu_fast_approximate(x),
        }
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the hyperbolic tangent function
#[derive(Debug, Clone, ModuleParameters)]
pub struct Tanh;

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Tanh {
    /// Creates a new [`Tanh`] module.
    /// 
    /// This is just to be consistent with the other modules.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Tanh {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        Ok(mlx_rs::ops::tanh(x))
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the hardswish function, element-wise
///
/// This is:
///
/// ```rust, ignore
/// x * minimum(maximum(x + 3, 0), 6) / 6
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct HardSwish;

impl Default for HardSwish {
    fn default() -> Self {
        Self::new()
    }
}

impl HardSwish {
    /// Creates a new [`HardSwish`] module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for HardSwish {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        hard_swish(x)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Step Activation Function.
///
/// This function implements a binary step activation, where the output is set
/// to 1 if the input is greater than a specified threshold, and 0 otherwise.
///
/// This is:
///
/// ```rust, ignore
/// r#where(x.gt(threshold), 1, 0)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Step {
    /// The threshold value. See [`step`] for more details.
    pub threshold: Option<f32>,
}

impl Default for Step {
    fn default() -> Self {
        Self::new()
    }
}

impl Step {
    /// Creates a new [`Step`] module.
    pub fn new() -> Self {
        Self { threshold: None }
    }

    /// Sets the value of the `threshold`
    pub fn with_threshold(mut self, threshold: impl Into<Option<f32>>) -> Self {
        self.threshold = threshold.into();
        self
    }
}

impl Module for Step {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        step(x, self.threshold)
    }

    fn train(&mut self, _: bool) {}
}

/// Applies the Scaled Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// elu(x, 1.67326) * 1.0507
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct Selu;

impl Default for Selu {
    fn default() -> Self {
        Self::new()
    }
}

impl Selu {
    /// Creates a new [`Selu`] module.
    /// 
    /// This is just to be consistent with the other modules.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Selu {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        selu(x)
    }

    fn train(&mut self, _: bool) {}
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
        mlx_rs::ops::which(
            &x_.gt(array!(0.0)).unwrap(),
            x_,
            alpha_ * (mlx_rs::ops::exp(x_) - array!(1.0)),
        )
        .unwrap()
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled((x, alpha))
}

#[inline]
fn compiled_relu6(x: &Array) -> Result<Array, Exception> {
    let f = |x_: &Array| {
        mlx_rs::ops::minimum(mlx_rs::ops::maximum(x_, array!(0.0)).unwrap(), array!(6.0)).unwrap()
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_softsign(x: &Array) -> Result<Array, Exception> {
    let f = |x_: &Array| x_ / (array!(1.0) + mlx_rs::ops::abs(x_));
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_celu(x: &Array, alpha: &Array) -> Result<Array, Exception> {
    let f = |(x_, alpha_): (&Array, &Array)| {
        mlx_rs::ops::maximum(x_, array!(0.0)).unwrap()
            + alpha_
                * (mlx_rs::ops::exp(&(mlx_rs::ops::minimum(x_, 0.0).unwrap() / alpha_))
                    - array!(1.0))
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
    let f = |x_: &Array| x_ * (array!(1) + erf(&(x_ / array!(2f32.sqrt())))) / array!(2.0);
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
                + tanh(
                    &(sqrt(&array!(2.0 / PI))
                        * (x_ + array!(0.044715) * x_.power(array!(3)).unwrap())),
                ))
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
    let f = |x_: &Array| elu(x_, 1.67326).unwrap() * array!(1.0507);
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[inline]
fn compiled_prelu(x: &Array, alpha: &Array) -> Result<Array, Exception> {
    let f = |(x_, alpha_): (&Array, &Array)| {
        mlx_rs::ops::maximum(array!(0.0), x_).unwrap()
            + alpha_ * mlx_rs::ops::minimum(array!(0.0), x_).unwrap()
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
        let max_x_plus_3 = mlx_rs::ops::maximum(&(x_ + array!(3.0)), array!(0.0)).unwrap();
        x_ * mlx_rs::ops::minimum(&max_x_plus_3, array!(6.0)).unwrap() / array!(6.0)
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let x = array!([-1.0, 0.0, 1.0]);
        let y = relu(&x).unwrap();
        assert_eq!(y, array!([0.0, 0.0, 1.0]));
    }
}
