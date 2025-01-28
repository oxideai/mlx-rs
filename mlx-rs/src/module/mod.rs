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

cfg_safetensors! {
    use crate::error::ModuleIoError;

    /// Load module parameters from a `safetensors` file.
    pub fn load_safetensors<M, P>(module: &mut M, path: P) -> Result<(), ModuleIoError>
    where
        M: ModuleParameters,
        P: AsRef<std::path::Path>,
    {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let mmap = unsafe {
            memmap2::Mmap::map(&file)?
        };
        let st = safetensors::SafeTensors::deserialize(&mmap[..])?;

        // Load the parameters
        let params = module.parameters_mut().flatten();
        for (key, value) in params {
            let tensor = st.tensor(&*key)?;
            *value = crate::Array::try_from(tensor)?;
        }
    
        Ok(())
    }

    /// Save module parameters to a file in `safetensors` format.
    pub fn save_safetensors<M, P>(module: &M, path: P) -> Result<(), ModuleIoError> 
    where
        M: ModuleParameters,
        P: AsRef<std::path::Path>,
    {
        let params = module.parameters().flatten();
        let iter = params.into_iter().map(|(k, v)| {
            let tensor = safetensors::tensor::TensorView::try_from(v)?;
            Result::<_, crate::error::ConversionError>::Ok((k, tensor))
        })
            .collect::<Result<Vec<_>, _>>()?;
        safetensors::tensor::serialize_to_file(iter, &None, path.as_ref())?;
        Ok(())
    }
}
