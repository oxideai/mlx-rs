use std::sync::Arc;

use mlx_internal_macros::generate_builder;
use mlx_macros::ModuleParameters;
use mlx_rs::{array, error::Exception, module::{Module, Param}, ops::{addmm, matmul, sigmoid, split, split_equal, stack, tanh, tanh_device}, prelude::{Ellipsis, IndexOp}, random::uniform, Array, Stream};

/// Type alias for the non-linearity function.
pub type NonLinearity = dyn Fn(&Array, &Stream) -> Result<Array, Exception>;


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
    pub bias: Param<Option<Array>>,
}

/// Builder for the [`Rnn`] module.
#[derive(Clone, Default)]
pub struct RnnBuilder {
    /// non-linearity function to use
    pub non_linearity: Option<Arc<NonLinearity>>,

    /// Bias. Default to [`Rnn::DEFAULT_BIAS`].
    pub bias: Option<bool>,
}

impl std::fmt::Debug for RnnBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("RnnBuilder")
            .field("bias", &self.bias)
            .finish()
    }
}

impl RnnBuilder {
    /// Create a new [`RnnBuilder`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the non-linearity function to use.
    pub fn non_linearity(mut self, non_linearity: impl Into<Option<Arc<NonLinearity>>>) -> Self {
        self.non_linearity = non_linearity.into();
        self
    }

    /// Set the bias.
    pub fn bias(mut self, bias: impl Into<Option<bool>>) -> Self {
        self.bias = bias.into();
        self
    }

    /// Build the [`Rnn`] module.
    pub fn build(
        self, 
        input_size: i32, 
        hidden_size: i32,
    ) -> Result<Rnn, Exception> {
        let non_linearity = self.non_linearity.unwrap_or_else(|| {
            Arc::new(|x, d| Ok(tanh_device(x, d)))
        });

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

    /// Create a new [`RnnBuilder`].
    pub fn builder() -> RnnBuilder {
        RnnBuilder::new()
    }

    /// Create a new [`RNN`] layer.
    pub fn new(
        input_size: i32,
        hidden_size: i32,
    ) -> Result<Rnn, Exception> {
        RnnBuilder::default().build(input_size, hidden_size)
    }

    fn step(&mut self, x: &Array, hidden: Option<&Array>) -> Result<Array, Exception> {
        let x = if let Some(bias) = &self.bias.value {
            addmm(bias, x, self.wxh.t(), None, None)?
        } else {
            matmul(x, &self.wxh.t())?
        };

        let mut all_hidden = Vec::new();
        for index in 0..x.dim(-2) {
            let hidden = match hidden {
                Some(hidden_) => addmm(x.index((Ellipsis, index, 0..)), hidden_, self.whh.t(), None, None)?,
                None => x.index((Ellipsis, index, 0..)),
            };

            let hidden = (self.non_linearity)(&hidden, &Stream::default())?;
            all_hidden.push(hidden);
        }

        stack(&all_hidden[..], -2)
    }
}

generate_builder! {
    /// A gated recurrent unit (GRU) RNN layer.
    ///
    /// The input has shape `NLD` or `LD` where:
    ///
    /// * `N` is the optional batch dimension
    /// * `L` is the sequence length
    /// * `D` is the input's feature dimension
    ///
    /// The hidden state `h` has shape `NH` or `H`, depending on
    /// whether the input is batched or not. Returns the hidden state at each
    /// time step, of shape `NLH` or `LH`.
    #[derive(Debug, Clone, ModuleParameters)]
    #[generate_builder(generate_build_fn = false)]
    pub struct Gru {
        /// Dimension of the hidden state, `H`
        pub hidden_size: i32,

        /// Wx
        #[param]
        pub wx: Param<Array>,

        /// Wh
        #[param]
        pub wh: Param<Array>,

        /// Bias. Enabled by default.
        #[param]
        #[optional(ty = bool)]
        pub bias: Param<Option<Array>>,

        /// bhn. Enabled by default.
        #[param]
        pub bhn: Param<Option<Array>>,
    }
}

impl GruBuilder {
    /// Build the [`Gru`] module.
    /// 
    /// # Params
    /// 
    /// - `input_size`: dimension of the input, `D` (see [`Gru`] for more details)
    /// - `hidden_size`: dimension of the hidden state, `H` (see [`Gru`] for more details)
    pub fn build(self, input_size: i32, hidden_size: i32) -> Result<Gru, Exception> {
        let scale = 1.0 / f32::sqrt(hidden_size as f32);
        let wx = uniform::<_, f32>(-scale, scale, &[3 * hidden_size, input_size], None)?;
        let wh = uniform::<_, f32>(-scale, scale, &[3 * hidden_size, input_size], None)?;
        let (bias, bhn) = if self.bias.unwrap_or(Gru::DEFAULT_BIAS) {
            let bias = uniform::<_, f32>(-scale, scale, &[3 * hidden_size], None)?;
            let bhn = uniform::<_, f32>(-scale, scale, &[hidden_size], None)?;
            (Some(bias), Some(bhn))
        } else {
            (None, None)
        };

        Ok(Gru {
            hidden_size,
            wx: Param::new(wx),
            wh: Param::new(wh),
            bias: Param::new(bias),
            bhn: Param::new(bhn),
        })
    }
}

impl Gru {
    /// Enable `bias` and `bhn` by default
    pub const DEFAULT_BIAS: bool = true;

    /// Create a new [`Gru`] layer.
    pub fn new(input_size: i32, hidden_size: i32) -> Result<Gru, Exception> {
        GruBuilder::default().build(input_size, hidden_size)
    }

    fn step(&mut self, x: &Array, hidden: Option<&Array>) -> Result<Array, Exception> {
        let x = if let Some(b) = &self.bias.value {
            addmm(b, x, self.wx.t(), None, None)?
        } else {
            matmul(x, &self.wx.t())?
        };

        let x_rz = x.index((Ellipsis, ..(-self.hidden_size)));
        let x_n = x.index((Ellipsis, (-self.hidden_size)..));

        let mut all_hidden = Vec::new();

        for index in 0..x.dim(-2) {
            let mut rz = x_rz.index((Ellipsis, index, 0..));
            let mut h_proj_n = None;
            if let Some(hidden_) = hidden {
                let h_proj = matmul(hidden_, &self.wh.t())?;
                let h_proj_rz = h_proj.index((Ellipsis, 0..(-self.hidden_size)));
                h_proj_n = Some(h_proj.index((Ellipsis, (-self.hidden_size)..)));

                if let Some(bhn) = &self.bhn.value {
                    h_proj_n = h_proj_n.map(|h_proj_n| h_proj_n.add(bhn))
                        // This is not matrix transpose, but from `Option<Result<_>>` to `Result<Option<_>>`
                        .transpose()?;
                }

                rz = rz.add(h_proj_rz)?;

            }
            
            rz = sigmoid(&rz);

            let parts = split_equal(&rz, 2, -1)?;
            let r = &parts[0];
            let z = &parts[1];

            let mut n = x_n.index((Ellipsis, index, 0..));

            if let Some(h_proj_n) = h_proj_n {
                n = n.add(r.multiply(h_proj_n)?)?;
            }
            n = tanh(&n);

            let hidden = match hidden {
                Some(hidden) => {
                    array!(1.0)
                        .subtract(z)?
                        .multiply(&n)?
                        .add(z.multiply(hidden)?)?
                },
                None => {
                    array!(1.0)
                        .subtract(z)?
                        .multiply(&n)?
                },
            };

            all_hidden.push(hidden);
        }

        stack(&all_hidden[..], -2)
    }
}

generate_builder! {
    /// A long short-term memory (LSTM) RNN layer.
    #[derive(Debug, Clone, ModuleParameters)]
    #[generate_builder(generate_build_fn = false)]
    pub struct Lstm {
        /// Wx
        #[param]
        pub wx: Param<Array>,

        /// Wh
        #[param]
        pub wh: Param<Array>,

        /// Bias. Enabled by default.
        #[param]
        #[optional(ty = bool)]
        pub bias: Param<Option<Array>>,
    }
}

impl LstmBuilder {
    /// Build the [`Lstm`] module.
    pub fn build(self, input_size: i32, hidden_size: i32) -> Result<Lstm, Exception> {
        let scale = 1.0 / f32::sqrt(hidden_size as f32);
        let wx = uniform::<_, f32>(-scale, scale, &[4 * hidden_size, input_size], None)?;
        let wh = uniform::<_, f32>(-scale, scale, &[4 * hidden_size, input_size], None)?;
        let bias = if self.bias.unwrap_or(Lstm::DEFAULT_BIAS) {
            Some(uniform::<_, f32>(-scale, scale, &[4 * hidden_size], None)?)
        } else {
            None
        };

        Ok(Lstm {
            wx: Param::new(wx),
            wh: Param::new(wh),
            bias: Param::new(bias),
        })
    }
}

impl Lstm {
    /// Default value for `bias`
    pub const DEFAULT_BIAS: bool = true;

    /// Create a new [`Lstm`] layer.
    pub fn new(input_size: i32, hidden_size: i32) -> Result<Lstm, Exception> {
        LstmBuilder::default().build(input_size, hidden_size)
    }

    fn step(&mut self, x: &Array, hidden: Option<&Array>, cell: Option<&Array>) -> Result<(Array, Array), Exception> {
        let x = if let Some(b) = &self.bias.value {
            addmm(b, x, self.wx.t(), None, None)?
        } else {
            matmul(x, &self.wx.t())?
        };

        let mut all_hidden = Vec::new();
        let mut all_cell = Vec::new();

        for index in 0..x.dim(-2) {
            let mut ifgo = x.index((Ellipsis, index, 0..));
            if let Some(hidden) = hidden {
                ifgo = addmm(&ifgo, hidden, self.wh.t(), None, None)?;
            }

            let pieces = split_equal(&ifgo, 4, -1)?;

            let i = sigmoid(&pieces[0]);
            let f = sigmoid(&pieces[1]);
            let g = tanh(&pieces[2]);
            let o = sigmoid(&pieces[3]);

            let cell = match cell {
                Some(cell) => {
                    f.multiply(cell)?.add(i.multiply(&g)?)?
                },
                None => {
                    i.multiply(&g)?
                },
            };

            let hidden = o.multiply(&tanh(&cell))?;

            all_hidden.push(hidden);
            all_cell.push(cell);
        }


        Ok((stack(&all_hidden[..], -2)?, stack(&all_cell[..], -2)?))
    }
}

#[cfg(test)]
mod tests {

    // def test_rnn(self):
    //     layer = nn.RNN(input_size=5, hidden_size=12, bias=True)
    //     inp = mx.random.normal((2, 25, 5))

    //     h_out = layer(inp)
    //     self.assertEqual(h_out.shape, (2, 25, 12))

    //     layer = nn.RNN(
    //         5,
    //         12,
    //         bias=False,
    //         nonlinearity=lambda x: mx.maximum(0, x),
    //     )

    //     h_out = layer(inp)
    //     self.assertEqual(h_out.shape, (2, 25, 12))

    //     with self.assertRaises(ValueError):
    //         nn.RNN(5, 12, nonlinearity="tanh")

    //     inp = mx.random.normal((44, 5))
    //     h_out = layer(inp)
    //     self.assertEqual(h_out.shape, (44, 12))

    //     h_out = layer(inp, hidden=h_out[-1, :])
    //     self.assertEqual(h_out.shape, (44, 12))

    use mlx_rs::{ops::maximum_device, random::normal};

    use super::*;

    #[test]
    fn test_rnn() {
        let mut layer = Rnn::new(5, 12).unwrap();
        let inp = normal::<f32>(&[2, 25, 5], None, None, None).unwrap();

        let h_out = layer.step(&inp, None).unwrap();
        assert_eq!(h_out.shape(), &[2, 25, 12]);

        let nonlinearity = |x: &Array, d: &Stream| maximum_device(x, array!(0.0), d);
        let mut layer = Rnn::builder()
            .bias(false)
            .non_linearity(Arc::new(nonlinearity) as Arc<NonLinearity>)
            .build(5, 12)
            .unwrap();

        let h_out = layer.step(&inp, None).unwrap();
        assert_eq!(h_out.shape(), &[2, 25, 12]);

        todo!()
    }
}