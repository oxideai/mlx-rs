use mlx_nn::{macros::ModuleParameters, Linear, Relu, Sequential};
use mlx_rs::{error::Exception, module::Module, Array};

#[derive(Debug, ModuleParameters)]
pub struct Mlp {
    #[param]
    pub layers: Sequential,
}

impl Module<&Array> for Mlp {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        self.layers.forward(x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.layers.training_mode(mode);
    }
}

impl Mlp {
    pub fn new(
        num_layers: usize,
        input_dim: i32,
        hidden_dim: i32,
        output_dim: i32,
    ) -> Result<Self, Exception> {
        let mut layers = Sequential::new();

        // Add the input layer
        layers = layers
            .append(Linear::new(input_dim, hidden_dim)?)
            .append(Relu);

        // Add the hidden layers
        for _ in 1..num_layers {
            layers = layers
                .append(Linear::new(hidden_dim, hidden_dim)?)
                .append(Relu);
        }

        // Add the output layer
        layers = layers.append(Linear::new(hidden_dim, output_dim)?);

        Ok(Self { layers })
    }
}
