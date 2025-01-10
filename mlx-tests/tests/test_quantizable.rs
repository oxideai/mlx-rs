use mlx_rs::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn::Linear,
    quantizable::MaybeQuantized,
    Array,
};

#[derive(Debug, ModuleParameters, Quantizable)]
struct QuantizableExample {
    #[quantizable]
    pub ql: MaybeQuantized<Linear>,
}

impl<'a> Module<'a> for QuantizableExample {
    type Input = &'a Array;

    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, x: Self::Input) -> Result<Self::Output, Self::Error> {
        self.ql.forward(x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.ql.training_mode(mode)
    }
}
