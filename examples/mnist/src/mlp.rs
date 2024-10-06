use mlx_nn::{macros::ModuleParameters, module::Param, Linear};

#[derive(Debug, ModuleParameters)]
pub struct Mlp {
    #[param]
    layers: Param<Vec<Linear>>
}