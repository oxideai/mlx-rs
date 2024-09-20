use std::f32::consts::PI;

use mlx_macros::ModuleParameters;
use mlx_nn_module::{Module, Param};
use mlx_rs::{
    array,
    error::Exception,
    ops::{log_sum_exp, multiply},
    transforms::compile::compile,
    Array,
};

use crate::error::Error;

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
    mlx_rs::ops::log_add_exp(x.as_ref(), &array!(0))
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
    mlx_rs::ops::r#where(&x.as_ref().gt(threshold)?, &array!(1), &array!(0))
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
    /// The axis to split the input tensor. Default to [`Glu::DEFAULT_AXIS`] if not provided.
    pub axis: i32,
}

impl Default for Glu {
    fn default() -> Self {
        Self::new()
    }
}

impl Glu {
    /// The default axis value.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Creates a [`Glu`] module.
    ///
    /// # Params
    ///
    /// - `axis`: The axis to split the input tensor. Default to -1 if not provided.
    pub fn new() -> Self {
        Self {
            axis: Self::DEFAULT_AXIS,
        }
    }

    /// Sets the value of the `axis` field. Default to [`Glu::DEFAULT_AXIS`] if not provided.
    pub fn with_axis(mut self, axis: impl Into<Option<i32>>) -> Self {
        self.axis = axis.into().unwrap_or(Self::DEFAULT_AXIS);
        self
    }
}

impl Module for Glu {
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        glu(x, self.axis).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        Ok(sigmoid(x))
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        mish(x).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        relu(x).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Leaky Rectified Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// maximum(neg_slope * x, x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
pub struct LeakyRelu {
    /// The negative slope. Default to [`LeakyReLU::`] if not provided.
    pub neg_slope: f32,
}

impl Default for LeakyRelu {
    fn default() -> Self {
        Self::new()
    }
}

impl LeakyRelu {
    /// The default negative slope value.
    pub const DEFAULT_NEG_SLOPE: f32 = 0.01;

    /// Creates a new [`LeakyReLU`] module.
    pub fn new() -> Self {
        Self {
            neg_slope: Self::DEFAULT_NEG_SLOPE,
        }
    }

    /// Sets the value of the `neg_slope`
    pub fn with_neg_slope(mut self, neg_slope: impl Into<Option<f32>>) -> Self {
        self.neg_slope = neg_slope.into().unwrap_or(Self::DEFAULT_NEG_SLOPE);
        self
    }
}

impl Module for LeakyRelu {
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        leaky_relu(x, self.neg_slope).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        relu6(x).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    pub axis: i32,
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Softmax {
    /// The default axis value.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Creates a new [`Softmax`] module.
    pub fn new() -> Self {
        Self {
            axis: Self::DEFAULT_AXIS,
        }
    }

    /// Sets the value of the `axis`
    pub fn with_axis(mut self, axis: impl Into<Option<i32>>) -> Self {
        self.axis = axis.into().unwrap_or(Self::DEFAULT_AXIS);
        self
    }
}

impl Module for Softmax {
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        Ok(mlx_rs::ops::softmax(x, &[self.axis], None))
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        softplus(x).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        softsign(x).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    /// The alpha value. Default to [`Celu::DEFAULT_ALPHA`] if not provided.
    pub alpha: f32,
}

impl Default for Celu {
    fn default() -> Self {
        Self::new()
    }
}

impl Celu {
    /// The default alpha value.
    pub const DEFAULT_ALPHA: f32 = 1.0;

    /// Creates a new [`Celu`] module.
    pub fn new() -> Self {
        Self {
            alpha: Self::DEFAULT_ALPHA,
        }
    }

    /// Sets the value of the `alpha`. Default to [`Celu::DEFAULT_ALPHA`] if not provided.
    pub fn with_alpha(mut self, alpha: impl Into<Option<f32>>) -> Self {
        self.alpha = alpha.into().unwrap_or(Self::DEFAULT_ALPHA);
        self
    }
}

impl Module for Celu {
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        celu(x, self.alpha).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        silu(x).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    /// The axis value. Default to [`LogSoftmax::DEFAULT_AXIS`] if not provided.
    pub axis: i32,
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSoftmax {
    /// The default axis value.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Creates a new [`LogSoftmax`] module.
    pub fn new() -> Self {
        Self {
            axis: Self::DEFAULT_AXIS,
        }
    }

    /// Sets the value of the `axis`. Default to [`LogSoftmax::DEFAULT_AXIS`] if not provided.
    pub fn with_axis(mut self, axis: impl Into<Option<i32>>) -> Self {
        self.axis = axis.into().unwrap_or(Self::DEFAULT_AXIS);
        self
    }
}

impl Module for LogSoftmax {
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        log_softmax(x, self.axis).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        log_sigmoid(x).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    pub weight: Param<Array>, // TODO: double check if this is trainable
}

impl Prelu {
    /// The default count value.
    pub const DEFAULT_COUNT: i32 = 1;

    /// The default value.
    pub const DEFAULT_VALUE: f32 = 0.25;

    /// Creates a new [`Prelu`] module.
    pub fn new() -> Self {
        let weight = mlx_rs::ops::full::<f32>(&[Self::DEFAULT_COUNT], &array!(Self::DEFAULT_VALUE))
            .expect("Creating the default weight for Prelu should not fail");
        Self {
            weight: Param::new(weight),
        }
    }

    /// Sets the value of the `weight` with the given `count` and `value`.
    pub fn with_count_and_value(
        mut self,
        count: impl Into<Option<i32>>,
        value: impl Into<Option<f32>>,
    ) -> Result<Self, Exception> {
        let count = count.into().unwrap_or(Self::DEFAULT_COUNT);
        let value = value.into().unwrap_or(Self::DEFAULT_VALUE);
        self.weight = Param::new(mlx_rs::ops::full::<f32>(&[count], &array!(value))?);
        Ok(self)
    }
}

impl Default for Prelu {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Prelu {
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        prelu(x, &self.weight).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        match self.approximate {
            GeluApprox::None => gelu(x).map_err(Into::into),
            GeluApprox::Precise => gelu_approximate(x).map_err(Into::into),
            GeluApprox::Fast => gelu_fast_approximate(x).map_err(Into::into),
        }
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        Ok(mlx_rs::ops::tanh(x))
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        hard_swish(x).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    /// The threshold value. Default to [`Step::DEFAULT_THRESHOLD`] if not provided.
    pub threshold: f32,
}

impl Default for Step {
    fn default() -> Self {
        Self::new()
    }
}

impl Step {
    /// The default threshold value.
    pub const DEFAULT_THRESHOLD: f32 = 0.0;

    /// Creates a new [`Step`] module.
    pub fn new() -> Self {
        Self {
            threshold: Self::DEFAULT_THRESHOLD,
        }
    }

    /// Sets the value of the `threshold`
    pub fn with_threshold(mut self, threshold: impl Into<Option<f32>>) -> Self {
        self.threshold = threshold.into().unwrap_or(Self::DEFAULT_THRESHOLD);
        self
    }
}

impl Module for Step {
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        step(x, self.threshold).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
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
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        selu(x).map_err(Into::into)
    }

    fn training_mode(&mut self, _: bool) {}
}

/* -------------------------------------------------------------------------- */
/*                        Compiled activation functions                       */
/* -------------------------------------------------------------------------- */

#[inline]
fn compiled_leaky_relu(x: &Array, neg_slope: &Array) -> Result<Array, Exception> {
    let f = |(x_, neg_slope_): (&Array, &Array)| {
        // This will not panic because a scalar can always be broadcasted to any shape
        let a = multiply(neg_slope_, x_).unwrap();
        mlx_rs::ops::maximum(&a, x_).unwrap()
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled((x, neg_slope))
}

#[inline]
fn compiled_elu(x: &Array, alpha: &Array) -> Result<Array, Exception> {
    let f = |(x_, alpha_): (&Array, &Array)| {
        mlx_rs::ops::which(
            &x_.gt(&array!(0.0)).unwrap(),
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
        mlx_rs::ops::minimum(
            &mlx_rs::ops::maximum(x_, &array!(0.0)).unwrap(),
            &array!(6.0),
        )
        .unwrap()
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
        mlx_rs::ops::maximum(x_, &array!(0.0)).unwrap()
            + alpha_
                * (mlx_rs::ops::exp(&(mlx_rs::ops::minimum(x_, &array!(0.0)).unwrap() / alpha_))
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
                        * (x_ + array!(0.044715) * x_.power(&array!(3)).unwrap())),
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
        mlx_rs::ops::maximum(&array!(0.0), x_).unwrap()
            + alpha_ * mlx_rs::ops::minimum(&array!(0.0), x_).unwrap()
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
        let max_x_plus_3 = mlx_rs::ops::maximum(&(x_ + array!(3.0)), &array!(0.0)).unwrap();
        x_ * mlx_rs::ops::minimum(&max_x_plus_3, &array!(6.0)).unwrap() / &array!(6.0)
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled(x)
}

// The following tests are ported from the swift binding:
// mlx-swift/Tests/MLXTests/IntegrationTests.swift
#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use mlx_rs::{random::uniform, Dtype};

    use super::*;

    #[test]
    fn test_glu() {
        mlx_rs::random::seed(850);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.547_252_66,
            abs <= 0.010_945_053
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            140.096_68,
            abs <= 2.801_933_5
        );
        let result = Glu::default().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 8]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.333_276_75,
            abs <= 0.006_665_535
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            42.659_424,
            abs <= 0.853_188_46
        );
    }

    #[test]
    fn test_sigmoid() {
        mlx_rs::random::seed(589);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.529_697_9,
            abs <= 0.010_593_958
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            135.602_66,
            abs <= 2.712_053_3
        );
        let result = Sigmoid.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.627_014,
            abs <= 0.012_540_28
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            160.515_58,
            abs <= 3.210_311_7
        );
    }

    #[test]
    fn test_mish() {
        mlx_rs::random::seed(122);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.501_719_8,
            abs <= 0.010_034_395
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            128.440_26,
            abs <= 2.568_805_2
        );
        let result = Mish.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.395_375_73,
            abs <= 0.007_907_514
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            101.216_19,
            abs <= 2.024_323_7
        );
    }

    #[test]
    fn test_relu() {
        mlx_rs::random::seed(400);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.478_322_74,
            abs <= 0.009_566_455
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            122.450_62,
            abs <= 2.449_012_5
        );
        let result = Relu.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.478_322_74,
            abs <= 0.009_566_455
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            122.450_62,
            abs <= 2.449_012_5
        );
    }

    #[test]
    fn test_leaky_relu() {
        mlx_rs::random::seed(93);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.499_930_68,
            abs <= 0.009_998_614
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            127.982_254,
            abs <= 2.559_645_2
        );
        let result = LeakyRelu::default().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.499_930_68,
            abs <= 0.009_998_614
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            127.982_254,
            abs <= 2.559_645_2
        );
    }

    #[test]
    fn test_relu6() {
        mlx_rs::random::seed(379);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.493_258_66,
            abs <= 0.009_865_173
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            126.274_216,
            abs <= 2.525_484_3
        );
        let result = Relu6.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.493_258_66,
            abs <= 0.009_865_173
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            126.274_216,
            abs <= 2.525_484_3
        );
    }

    #[test]
    fn test_softmax() {
        mlx_rs::random::seed(853);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.514_396_3,
            abs <= 0.010_287_926_5
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            131.685_46,
            abs <= 2.633_709_2
        );
        let result = Softmax::default().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.062_499_996,
            abs <= 0.001_25
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            15.999_999,
            abs <= 0.32
        );
    }

    #[test]
    fn test_softplus() {
        mlx_rs::random::seed(118);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.498_981_42,
            abs <= 0.009_979_628
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            127.739_24,
            abs <= 2.554_784_8
        );
        let result = Softplus.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.982_857_76,
            abs <= 0.019_657_155
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            251.611_59,
            abs <= 5.032_232
        );
    }

    #[test]
    fn test_softsign() {
        mlx_rs::random::seed(37);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.506_551_27,
            abs <= 0.010_131_026
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            129.677_12,
            abs <= 2.593_542_6
        );
        let result = Softsign.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.314_089_83,
            abs <= 0.006_281_797
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            80.407,
            abs <= 1.608_14
        );
    }

    #[test]
    fn test_celu() {
        mlx_rs::random::seed(620);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.466_748_18,
            abs <= 0.009_334_964
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            119.487_53,
            abs <= 2.389_750_7
        );
        let result = Celu::default().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.466_748_18,
            abs <= 0.009_334_964
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            119.487_53,
            abs <= 2.389_750_7
        );
    }

    #[test]
    fn test_silu() {
        mlx_rs::random::seed(22);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.502_970_6,
            abs <= 0.010_059_412
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            128.760_47,
            abs <= 2.575_209_4
        );
        let result = Silu.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.331_970_93,
            abs <= 0.006_639_418_7
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            84.984_56,
            abs <= 1.699_691_2
        );
    }

    #[test]
    fn test_log_softmax() {
        mlx_rs::random::seed(199);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.527_843_7,
            abs <= 0.010_556_874
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            135.127_99,
            abs <= 2.702_559_7
        );
        let result = LogSoftmax::default().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            -2.810_954_6,
            abs <= 0.056_219_09
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            -719.604_4,
            abs <= 14.392_087
        );
    }

    #[test]
    fn test_log_sigmoid() {
        mlx_rs::random::seed(984);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.510_977_7,
            abs <= 0.010_219_553_5
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            130.810_29,
            abs <= 2.616_205_7
        );
        let result = LogSigmoid.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            -0.479_598_55,
            abs <= 0.009_591_971
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            -122.777_23,
            abs <= 2.455_544_5
        );
    }

    #[test]
    fn test_prelu() {
        mlx_rs::random::seed(993);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.496_651_44,
            abs <= 0.009_933_028
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            127.142_77,
            abs <= 2.542_855_3
        );
        let result = Prelu::default().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.496_651_44,
            abs <= 0.009_933_028
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            127.142_77,
            abs <= 2.542_855_3
        );
    }

    #[test]
    fn test_gelu() {
        mlx_rs::random::seed(189);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.492_950_32,
            abs <= 0.009_859_007
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            126.195_28,
            abs <= 2.523_905_8
        );
        let result = Gelu::default().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.365_638_38,
            abs <= 0.007_312_767_7
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            93.603_424,
            abs <= 1.872_068_5
        );
    }

    #[test]
    fn test_tanh() {
        mlx_rs::random::seed(735);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.474_122_7,
            abs <= 0.009_482_454_5
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            121.375_41,
            abs <= 2.427_508_4
        );
        let result = Tanh.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.413_079_68,
            abs <= 0.008_261_594
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            105.748_4,
            abs <= 2.114_968
        );
    }

    #[test]
    fn test_hardswish() {
        mlx_rs::random::seed(126);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.491_892_46,
            abs <= 0.009_837_849
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            125.924_47,
            abs <= 2.518_489_4
        );
        let result = HardSwish.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.299_602_24,
            abs <= 0.005_992_044_7
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            76.698_17,
            abs <= 1.533_963_4
        );
    }

    #[test]
    fn test_step() {
        mlx_rs::random::seed(490);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.479_360_64,
            abs <= 0.009_587_212_5
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            122.716_324,
            abs <= 2.454_326_4
        );
        let result = Step::default().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Int32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            1.0,
            abs <= 0.02
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            256.0,
            abs <= 5.12
        );
    }

    #[test]
    fn test_selu() {
        mlx_rs::random::seed(215);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.493_026_8,
            abs <= 0.009_860_536
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            126.214_86,
            abs <= 2.524_297_2
        );
        let result = Selu.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.518_023_2,
            abs <= 0.010_360_463_5
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            132.613_94,
            abs <= 2.652_278_7
        );
    }
}
