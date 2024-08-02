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

#[inline]
fn leaky_relu_inner(x: &Array, neg_slope: &Array) -> Result<Array, Exception> {
    let f = |(x_, neg_slope_): (&Array, &Array)| {
        // This will not panic because a scalar can always be broadcasted to any shape
        mlx_rs::ops::maximum(neg_slope_ * x_, x_).unwrap()
    };
    let mut compiled = compile(f, Some(true), None, None);
    compiled((x, neg_slope))
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
    // We have to use this inner function, otherwise the compiler cannot
    // infer the lifetime of the value returned by the closure properly
    leaky_relu_inner(x.as_ref(), &neg_slope)
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

pub fn glu(x: impl AsRef<Array>, axis: impl Into<Option<i32>>) -> Result<Array, Exception> {
    let split = x.as_ref().split_equal(2, axis)?;
    let (a, b) = (&split[0], &split[1]);
    Ok(a * sigmoid(b))
}
