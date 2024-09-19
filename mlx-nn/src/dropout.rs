use mlx_macros::ModuleParameters;
use mlx_nn_module::Module;
use mlx_rs::{array, ops::multiply, random::bernoulli};

use crate::error::{Dropout2dError, Dropout3dError, Error};

/// Randomly zero a portion of the elements during training.
///
/// The remaining elements are multiplied with `1 / (1-p)` where
/// `p` is the probability of zeroing an element. This is done so the
/// expected value of a given element will remain the same.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Dropout {
    /// `1-p`, where `p` is the probability of zeroing an element. `p` is default to
    /// [`Dropout::DEFAULT_P`] if not specified.
    pub one_minus_p: f32,

    /// Whether the layer is in training mode. Default to [`Dropout::DEFAULT_TRAINING`] if not
    /// specified.
    pub training: bool,
}

impl Dropout {
    /// Default value for the probability of zeroing an element.
    pub const DEFAULT_P: f32 = 0.5;

    /// Default value for the training mode.
    pub const DEFAULT_TRAINING: bool = true;

    /// Creates a new dropout layer with the default parameters.
    pub fn new() -> Self {
        Self {
            one_minus_p: 1.0 - Self::DEFAULT_P,
            training: Self::DEFAULT_TRAINING,
        }
    }

    /// Sets the probability of zeroing an element.
    /// 
    /// # Panics
    /// 
    /// Panics if `p` is not in the range `[0, 1)`.
    pub fn with_p(mut self, p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "p must be in the range [0, 1)");

        self.one_minus_p = 1.0 - p;
        self
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Dropout {
    type Error = Error;

    fn forward(&self, x: &mlx_rs::Array) -> Result<mlx_rs::Array, Self::Error> {
        if self.one_minus_p == 1.0 || !self.training {
            return Ok(x.clone());
        }

        let p1 = array!(self.one_minus_p);
        let mask = bernoulli(&p1, x.shape(), None)?;
        multiply(multiply(1.0 / self.one_minus_p, mask)?, x).map_err(Into::into)
    }

    fn training_mode(&mut self, mode: bool) {
        self.training = mode;
    }
}

/// Apply 2D channel-wise dropout during training.
///
/// Randomly zero out entire channels independently with probability `p`.
/// This layer expects the channels to be last, i.e. the input shape should be
/// `NWHC` or `WHC` where:`N` is the batch dimension,`H` is the input
/// image height,`W` is the input image width, and`C` is the number of
/// input channels
///
/// The remaining channels are scaled by `1 / (1-p)` to
/// maintain the expected value of each element. Unlike traditional dropout,
/// which zeros individual entries, this layer zeros entire channels. This is
/// beneficial for early convolution layers where adjacent pixels are
/// correlated. In such case, traditional dropout may not effectively
/// regularize activations. For more details, see [1].
///
/// [1]: Thompson, J., Goroshin, R., Jain, A., LeCun, Y. and Bregler C., 2015.
/// Efficient Object Localization Using Convolutional Networks. CVPR 2015.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Dropout2d {
    /// `1-p`, where `p` is the probability of zeroing a channel. `p` is default to
    /// [`Dropout2d::DEFAULT_P`] if not specified.
    pub one_minus_p: f32,

    /// Whether the layer is in training mode. Default to [`Dropout2d::DEFAULT_TRAINING`] if not
    /// specified. Default to [`Dropout2d::DEFAULT_TRAINING`] if not specified.
    pub training: bool,
}

impl Dropout2d {
    /// Default value for the probability of zeroing a channel.
    pub const DEFAULT_P: f32 = 0.5;

    /// Default value for the training mode.
    pub const DEFAULT_TRAINING: bool = true;

    /// Creates a new dropout layer with the default parameters.
    pub fn new() -> Self {
        Self {
            one_minus_p: 1.0 - Self::DEFAULT_P,
            training: Self::DEFAULT_TRAINING,
        }
    }

    /// Sets the probability of zeroing a channel.
    /// 
    /// # Panics
    /// 
    /// Panics if `p` is not in the range `[0, 1)`.
    pub fn with_p(mut self, p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "p must be in the range [0, 1)");

        self.one_minus_p = 1.0 - p;
        self
    }
}

impl Default for Dropout2d {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Dropout2d {
    type Error = Error;

    fn forward(&self, x: &mlx_rs::Array) -> Result<mlx_rs::Array, Self::Error> {
        let ndim = x.ndim();

        if ndim != 3 && ndim != 4 {
            return Err(Dropout2dError::NdimNotSupported.into());
        }

        if self.one_minus_p == 1.0 || !self.training {
            return Ok(x.clone());
        }

        // Dropout is applied on the whole channel
        // 3D input: (1, 1, C)
        // 4D input: (B, 1, 1, C)

        let mut mask_shape = x.shape().to_vec();
        let len = mask_shape.len();
        mask_shape[len-2] = 1;
        mask_shape[len-3] = 1;

        let p1 = array!(self.one_minus_p);
        let mask = bernoulli(&p1, &mask_shape, None)?;

        multiply(multiply(1.0 / self.one_minus_p, mask)?, x).map_err(Into::into)
    }

    fn training_mode(&mut self, mode: bool) {
        self.training = mode;
    }
}

/// Apply 3D channel-wise dropout during training.
///
/// Randomly zero out entire channels independently with probability `p`.
/// This layer expects the channels to be last, i.e., the input shape should be
/// `NDHWC` or `DHWC` where: `N` is the batch dimension, `D` is the depth,
/// `H` is the input image height, `W` is the input image width, and `C` is
/// the number of input channels.
///
/// The remaining channels are scaled by `1 / (1-p)` to
/// maintain the expected value of each element. Unlike traditional dropout,
/// which zeros individual entries, this layer zeros entire channels. This is
/// often beneficial for convolutional layers processing 3D data, like in
/// medical imaging or video processing.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Dropout3d {
    /// `1-p`, where `p` is the probability of zeroing a channel. `p` is default to
    /// [`Dropout3d::DEFAULT_P`] if not specified.
    pub one_minus_p: f32,

    /// Whether the layer is in training mode. Default to [`Dropout3d::DEFAULT_TRAINING`] if not
    /// specified.
    pub training: bool,
}

impl Dropout3d {
    /// Default value for the probability of zeroing a channel.
    pub const DEFAULT_P: f32 = 0.5;

    /// Default value for the training mode.
    pub const DEFAULT_TRAINING: bool = true;

    /// Creates a new dropout layer with the default parameters.
    pub fn new() -> Self {
        Self {
            one_minus_p: 1.0 - Self::DEFAULT_P,
            training: Self::DEFAULT_TRAINING,
        }
    }

    /// Sets the probability of zeroing a channel.
    /// 
    /// # Panics
    /// 
    /// Panics if `p` is not in the range `[0, 1)`.
    pub fn with_p(mut self, p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "p must be in the range [0, 1)");

        self.one_minus_p = 1.0 - p;
        self
    }
}

impl Default for Dropout3d {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Dropout3d {
    type Error = Error;
    
    fn forward(&self, x: &mlx_rs::Array) -> Result<mlx_rs::Array, Self::Error> {
        let ndim = x.ndim();

        if ndim != 4 && ndim != 5 {
            return Err(Dropout3dError::NdimNotSupported.into());
        }

        if self.one_minus_p == 1.0 || !self.training {
            return Ok(x.clone());
        }

        // Dropout is applied on the whole channel
        // 4D input: (1, 1, 1, C)
        // 5D input: (B, 1, 1, 1, C)

        let mut mask_shape = x.shape().to_vec();
        let len = mask_shape.len();
        mask_shape[len-2] = 1;
        mask_shape[len-3] = 1;
        mask_shape[len-4] = 1;

        let p1 = array!(self.one_minus_p);
        let mask = bernoulli(&p1, &mask_shape, None)?;

        multiply(multiply(1.0 / self.one_minus_p, mask)?, x).map_err(Into::into)
    }
    
    fn training_mode(&mut self, mode: bool) {
        self.training = mode;
    }
}