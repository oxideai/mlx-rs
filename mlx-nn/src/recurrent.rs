use std::sync::Arc;

use mlx_internal_macros::generate_builder;
use mlx_macros::ModuleParameters;
use mlx_rs::{error::Exception, module::{Module, Param}, ops::{addmm, matmul}, random::uniform, Array, Stream};

/// Type alias for the non-linearity function.
pub type NonLinearity = dyn Fn(&Array, &Stream) -> Result<Array, Exception>;

generate_builder! {
    /// An Elman recurrent layer.
    ///
    /// The input is a sequence of shape `NLD` or `LD` where:
    ///
    /// * `N` is the optional batch dimension
    /// * `L` is the sequence length
    /// * `D` is the input's feature dimension
    ///
    /// The hidden state `h` has shape `NH` or `H`, depending on
    /// whether the input is batched or not. Returns the hidden state at each
    /// time step, of shape `NLH` or `LH`.
    #[derive(Clone, ModuleParameters)]
    #[generate_builder(generate_build_fn = false)]
    pub struct Rnn {
        /// non-linearity function to use
        pub non_linearity: Arc<NonLinearity>,
    
        /// Wxh
        #[param]
        pub wxh: Param<Array>,
    
        /// Whh
        #[param]
        pub whh: Param<Array>,
    
        /// Bias. Enabled by default.
        #[param]
        #[optional(ty = bool)]
        pub bias: Param<Option<Array>>,
    }
}

impl RnnBuilder {
    /// Build the [`Rnn`] module.
    pub fn build(
        self, 
        input_size: i32, 
        hidden_size: i32,
        non_linearity: impl Fn(&Array, &Stream) -> Result<Array, Exception> + 'static
    ) -> Result<Rnn, Exception> {
        let non_linearity = Arc::new(non_linearity);

        let scale = 1.0 / (input_size as f32).sqrt();
        let wxh = uniform::<_, f32>(-scale, scale, &[hidden_size, input_size], None)?;
        let whh = uniform::<_, f32>(-scale, scale, &[hidden_size, hidden_size], None)?;
        let bias = if self.bias.unwrap_or(Rnn::DEFAULT_BIAS) {
            Some(uniform::<_, f32>(-scale, scale, &[hidden_size], None)?)
        } else {
            None
        };

        Ok(Rnn {
            non_linearity,
            wxh: Param::new(wxh),
            whh: Param::new(whh),
            bias: Param::new(bias),
        })
    }
}

impl std::fmt::Debug for Rnn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Rnn")
            .field("wxh", &self.wxh)
            .field("whh", &self.whh)
            .field("bias", &self.bias)
            .finish()
    }
}

impl Rnn {
    /// Default value for bias
    pub const DEFAULT_BIAS: bool = true;

    /// Create a new [`RNN`] layer.
    pub fn new(
        input_size: i32,
        hidden_size: i32,
        non_linearity: impl Fn(&Array, &Stream) -> Result<Array, Exception> + 'static
    ) -> Result<Rnn, Exception> {
        RnnBuilder::default().build(input_size, hidden_size, non_linearity)
    }

    fn forward_inner(&mut self, x: &Array, hidden: Option<&Array>) -> Result<Array, Exception> {

    }
}

// impl Module for Rnn {
//     type Error = Exception;

//     fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
//         let x = if let Some(bias) = &self.bias.value {
//             addmm(bias, x, self.wxh.t(), None, None)?
//         } else {
//             matmul(x, &self.wxh.t())?
//         };

//         todo!()
//     }

//     fn training_mode(&mut self, _mode: bool) {

//     }
// }