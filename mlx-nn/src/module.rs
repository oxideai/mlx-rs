use mlx_rs::{error::Exception, Array};

/// Type placeholder for module parameters.
pub type ModuleParameters = Vec<Array>;
pub type ModuleParametersRef<'a> = Vec<&'a Array>;
pub type ModuleParametersMut<'a> = Vec<&'a mut Array>;

pub trait Module {
    // TODO: Should we use `&Array` instead of `Array`? What if an op does nothing and just return
    // the same array?
    fn forward(&self, x: &Array) -> Result<Array, Exception>;

    fn parameters(&self) -> ModuleParametersRef<'_> {
        todo!("Remove default implementation and implement this method for each module");
    }

    fn parameters_mut(&mut self) -> ModuleParametersMut<'_> {
        todo!("Remove default implementation and implement this method for each module");
    }

    fn update(&mut self, parameters: ModuleParameters) {
        self.parameters_mut()
            .into_iter()
            .zip(parameters.into_iter())
            .for_each(|(param, new_param)| {
                *param = new_param;
            });
    }
}
