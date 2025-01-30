//! This mod defines the traits for neural network modules and parameters.
//!
//! This is to separate the trait definitions from the implementations, which are in the `mlx-nn`
//! crate. This also allows using the `mlx_macros::ModuleParameters` derive macro in crates other
//! than `mlx-nn`.

#[allow(clippy::module_inception)]
mod module;
mod param;

pub use module::*;
pub use param::*;

use crate::{
    error::{Exception, IoError},
    Array,
};

/// Extension trait for `ModuleParameters`. This is implemented for all types that implement
/// `ModuleParameters`.
pub trait ModuleParametersExt: ModuleParameters {
    /// Evaluate the module parameters.
    fn eval(&self) -> Result<(), Exception> {
        crate::transforms::eval_params(self.parameters())
    }

    /// Load module parameters from a `safetensors` file.
    fn load_safetensors<P>(&mut self, path: P) -> Result<(), IoError>
    where
        P: AsRef<std::path::Path>,
    {
        let weights = Array::load_safetensors(path)?;

        // Load the parameters
        let mut params = self.parameters_mut().flatten();
        for (key, value) in weights {
            if let Some(param) = params.get_mut(&*key) {
                **param = value;
            }
        }

        // Loading is lazy, eval after loading
        crate::transforms::eval_params(self.parameters())?;

        Ok(())
    }

    /// Save module parameters to a file in `safetensors` format.
    fn save_safetensors<M, P>(module: &M, path: P) -> Result<(), IoError>
    where
        M: ModuleParameters,
        P: AsRef<std::path::Path>,
    {
        let params = module.parameters().flatten();
        Array::save_safetensors(params, None, path)?;
        Ok(())
    }
}

impl<T: ModuleParameters> ModuleParametersExt for T {}
