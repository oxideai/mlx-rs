use std::{
    iter::{once, zip},
    sync::Arc,
};

use mlx_macros::ModuleParameters;
use mlx_rs::{error::Exception, module::Module, ops::as_strided, Array};

use crate::utils::SingleOrPair;

/// Type alias for a pooling operation.
type PoolingOp = dyn Fn(&Array, &[i32]) -> Result<Array, Exception>;

/// Abstract pooling layer.
///
/// See also:
///
/// - [`MaxPool1d`]
/// - [`MaxPool2d`]
/// - [`AvgPool1d`]
/// - [`AvgPool2d`]
#[derive(Clone, ModuleParameters)]
pub struct Pool {
    /// Size of the pooling window
    kernel_size: Vec<i32>,

    /// Stride of the pooling window
    stride: Vec<usize>,

    /// Axes to pool over
    axes: Vec<i32>,

    /// Pooling operation
    ///
    /// TODO: We have Arc here just to make it `Clone` and `Send`. Is this necessary?
    pooling_op: Arc<PoolingOp>,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("axes", &self.axes)
            .finish()
    }
}

impl Pool {
    /// Create a new abstract pooling layer.
    pub fn new(
        kernel_size: Vec<i32>,
        stride: Vec<usize>,
        op: impl Fn(&Array, &[i32]) -> Result<Array, Exception> + 'static,
    ) -> Self {
        let start = -(kernel_size.len() as i32) - 1;
        let axes: Vec<_> = (start..-1).collect();
        Self {
            kernel_size,
            stride,
            axes,
            pooling_op: Arc::new(op),
        }
    }
}

impl Module for Pool {
    type Error = Exception;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        let shape = x.shape();
        let rest = &shape[1..shape.len() - 1];

        let iter = zip(zip(rest, &self.kernel_size), &self.stride)
            .map(|((size, window), stride)| (size - window) / *stride as i32 + 1);

        let final_shape = once(shape[0])
            .chain(iter)
            .chain(self.kernel_size.iter().copied())
            .chain(once(shape[shape.len() - 1]))
            .collect::<Vec<_>>();

        let strides = shape
            .iter()
            .map(|s| *s as usize)
            .chain(once(1))
            .rev()
            .fold(vec![], |mut acc, a| {
                match acc.last() {
                    Some(&element) => acc.push(a * element),
                    None => acc.push(a),
                }
                acc
            })
            .into_iter()
            .rev()
            .skip(1)
            .collect::<Vec<_>>();
        let middle_strides = &strides[1..strides.len() - 1];

        let final_strides = once(strides[0])
            .chain(zip(middle_strides, &self.stride).map(|(ms, s)| ms * s))
            .chain(middle_strides.iter().copied())
            .chain(once(1))
            .collect::<Vec<_>>();

        // TODO: double check if as_strided would ever panic
        let strided = as_strided(x, &final_shape, &final_strides, None);
        (self.pooling_op)(&strided, &self.axes)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

macro_rules! impl_module {
    ($name:ident) => {
        impl Module for $name {
            type Error = Exception;

            fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
                self.inner.forward(x)
            }

            fn training_mode(&mut self, mode: bool) {
                self.inner.training_mode(mode);
            }
        }
    };
}

/// Applies 1-dimensional max pooling.
///
/// The input is expected to be `NLC`. The output will have the same N/C dimensions with the new `L
/// = floor((L - kernel)/stride) + 1`
///
/// See [MaxPool1d python
/// docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool1d.html)
/// for more information.
#[derive(Debug, Clone, ModuleParameters)]
pub struct MaxPool1d {
    inner: Pool,
}

impl MaxPool1d {
    /// Create a new 1-dimensional max pooling layer.
    ///
    /// # Params
    ///
    /// - `kernel_size`: The size of the pooling window.
    /// - `stride`: The stride of the pooling window.
    pub fn new(kernel_size: i32, stride: usize) -> Self {
        let op = |x: &Array, axes: &[i32]| x.max(axes, None);
        let inner = Pool::new(vec![kernel_size], vec![stride], op);
        Self { inner }
    }
}

impl_module!(MaxPool1d);

/// Applies 2-dimensional max pooling.
///
/// The input is expected to be `NHWC`. The output will have the same N/C dimensions with the new
/// `H/W = floor((H/W - kernel)/stride) + 1`
///
/// See [MaxPool2d python
/// docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool2d.html)
/// for more information.
#[derive(Debug, Clone, ModuleParameters)]
pub struct MaxPool2d {
    inner: Pool,
}

impl MaxPool2d {
    /// Create a new 2-dimensional max pooling layer.
    ///
    /// # Params
    ///
    /// - `kernel_size`: The size of the pooling window.
    /// - `stride`: The stride of the pooling window.
    pub fn new(kernel_size: impl SingleOrPair<i32>, stride: impl SingleOrPair<usize>) -> Self {
        let kernel_size = kernel_size.into_pair();
        let kernel_size = vec![kernel_size.0, kernel_size.1];
        let stride = stride.into_pair();
        let stride = vec![stride.0, stride.1];

        let op = |x: &Array, axes: &[i32]| x.max(axes, None);
        let inner = Pool::new(kernel_size, stride, op);
        Self { inner }
    }
}

impl_module!(MaxPool2d);

/// Applies 1-dimensional average pooling.
///
/// The input is expected to be `NLC`. The output will have the same N/C dimensions with the new `L =
/// floor((L - kernel)/stride) + 1`
///
/// See [AvgPool2d python
/// docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool2d.html)
/// for more information.
#[derive(Debug, Clone, ModuleParameters)]
pub struct AvgPool1d {
    inner: Pool,
}

impl AvgPool1d {
    /// Create a new 1-dimensional average pooling layer.
    ///
    /// # Params
    ///
    /// - `kernel_size`: The size of the pooling window.
    /// - `stride`: The stride of the pooling window.
    pub fn new(kernel_size: i32, stride: usize) -> Self {
        let op = |x: &Array, axes: &[i32]| x.mean(axes, None);
        let inner = Pool::new(vec![kernel_size], vec![stride], op);
        Self { inner }
    }
}

impl_module!(AvgPool1d);

/// Applies 2-dimensional average pooling.
///
/// The input is expected to be `NHWC`. The output will have the same N/C dimensions with the new
/// `H/W = floor((H/W - kernel)/stride) + 1`
///
/// See [AvgPool2d python
/// docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool2d.html)
/// for more information.
#[derive(Debug, Clone, ModuleParameters)]
pub struct AvgPool2d {
    inner: Pool,
}

impl AvgPool2d {
    /// Create a new 2-dimensional average pooling layer.
    ///
    /// # Params
    ///
    /// - `kernel_size`: The size of the pooling window.
    /// - `stride`: The stride of the pooling window.
    pub fn new(kernel_size: impl SingleOrPair<i32>, stride: impl SingleOrPair<usize>) -> Self {
        let kernel_size = kernel_size.into_pair();
        let kernel_size = vec![kernel_size.0, kernel_size.1];
        let stride = stride.into_pair();
        let stride = vec![stride.0, stride.1];

        let op = |x: &Array, axes: &[i32]| x.mean(axes, None);
        let inner = Pool::new(kernel_size, stride, op);
        Self { inner }
    }
}

impl_module!(AvgPool2d);

#[cfg(test)]
mod tests {
    use mlx_rs::{array, assert_array_eq};

    use super::*;

    #[test]
    fn test_max_pooling_1d_stride_1() {
        let input = Array::from_iter(0..4, &[1, 4, 1]);
        let pool = MaxPool1d::new(2, 1);
        let output = pool.forward(&input).unwrap();
        assert_array_eq!(output, array!([1, 2, 3], shape = [1, 3, 1]));
    }

    #[test]
    fn test_max_pooling_1d_stride_2() {
        let input = Array::from_iter(0..8, &[2, 4, 1]);
        let pool = MaxPool1d::new(2, 2);
        let output = pool.forward(&input).unwrap();
        assert_array_eq!(output, array!([1, 3, 5, 7], shape = [2, 2, 1]));
    }

    #[test]
    fn test_max_pooling_2d_stride_1() {
        let input = Array::from_iter(0..16, &[1, 4, 4, 1]);
        let pool = MaxPool2d::new(2, 1);
        let output = pool.forward(&input).unwrap();
        assert_array_eq!(
            output,
            array!([5, 6, 7, 9, 10, 11, 13, 14, 15], shape = [1, 3, 3, 1])
        );
    }

    #[test]
    fn test_max_pooling_2d_stride_2() {
        let input = Array::from_iter(0..32, &[2, 4, 4, 1]);
        let pool = MaxPool2d::new(2, 2);
        let output = pool.forward(&input).unwrap();
        assert_array_eq!(
            output,
            array!([5, 7, 13, 15, 21, 23, 29, 31], shape = [2, 2, 2, 1])
        );
    }

    #[test]
    fn test_avg_pooling_1d_stride_1() {
        let input = Array::from_iter(0..4, &[1, 4, 1]);
        let pool = AvgPool1d::new(2, 1);
        let output = pool.forward(&input).unwrap();
        assert_array_eq!(output, array!([0.5, 1.5, 2.5], shape = [1, 3, 1]));
    }

    #[test]
    fn test_avg_pooling_1d_stride_2() {
        let input = Array::from_iter(0..8, &[2, 4, 1]);
        let pool = AvgPool1d::new(2, 2);
        let output = pool.forward(&input).unwrap();
        assert_array_eq!(output, array!([0.5, 2.5, 4.5, 6.5], shape = [2, 2, 1]));
    }

    #[test]
    fn test_avg_pooling_2d_stride_1() {
        let input = Array::from_iter(0..16, &[1, 4, 4, 1]);
        let pool = AvgPool2d::new(2, 1);
        let output = pool.forward(&input).unwrap();
        assert_array_eq!(
            output,
            array!(
                [2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 10.5, 11.5, 12.5],
                shape = [1, 3, 3, 1]
            )
        );
    }

    #[test]
    fn test_avg_pooling_2d_stride_2() {
        let input = Array::from_iter(0..16, &[1, 4, 4, 1]);
        let pool = AvgPool2d::new(2, 2);
        let output = pool.forward(&input).unwrap();
        assert_array_eq!(output, array!([2.5, 4.5, 10.5, 12.5], shape = [1, 2, 2, 1]));
    }
}