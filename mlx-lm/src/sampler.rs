use mlx_rs::{Array, argmax_axis, array, categorical, error::Exception};

pub trait Sampler {
    fn sample(&mut self, logits: &Array, temp: f32) -> Result<Array, Exception>;
}

pub struct DefaultSampler;

impl Sampler for DefaultSampler {
    fn sample(&mut self, logits: &Array, temp: f32) -> Result<Array, Exception> {
        match temp {
            0.0 => argmax_axis!(logits, -1),
            _ => {
                let logits = logits.multiply(array!(1.0 / temp))?;
                categorical!(logits)
            }
        }
    }
}
