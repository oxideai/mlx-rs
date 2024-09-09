use std::{collections::HashMap, rc::Rc};

use mlx_rs::{error::Exception, nested::NestedHashMap, Array};

/// Type placeholder for module parameters.
pub type ModuleParam = NestedHashMap<&'static str, Array>;
pub type ModuleParamRef<'a> = NestedHashMap<&'static str, &'a Array>;
pub type ModuleParamMut<'a> = NestedHashMap<&'static str, &'a mut Array>;

pub type FlattenedModuleParam = HashMap<Rc<str>, Array>;
pub type FlattenedModuleParamRef<'a> = HashMap<Rc<str>, &'a Array>;
pub type FlattenedModuleParamMut<'a> = HashMap<Rc<str>, &'a mut Array>;

pub trait Module: ModuleParameters {
    fn forward(&self, x: &Array) -> Result<Array, Exception>;
}

pub trait ModuleParameters {
    fn parameters(&self) -> ModuleParamRef<'_>;
    fn parameters_mut(&mut self) -> ModuleParamMut<'_>;
    fn trainable_parameters(&self) -> ModuleParamRef<'_>;

    fn update(&mut self, parameters: ModuleParam) {
        let flattened_parameters = parameters.flatten();
        update_flattened_parameters(self, flattened_parameters)
    }

    fn update_flattened(&mut self, flattened_parameters: FlattenedModuleParam) {
        update_flattened_parameters(self, flattened_parameters)
    }
}

pub fn update_flattened_parameters<M, I>(module: &mut M, flattened_parameters: I)
where
    M: ModuleParameters + ?Sized,
    I: IntoIterator<Item = (Rc<str>, Array)>,
{
    let mut flattened_self_parameters = module.parameters_mut().flatten();

    for (key, value) in flattened_parameters {
        if let Some(self_value) = flattened_self_parameters.get_mut(&key) {
            **self_value = value.to_owned();
        }
    }
}

impl<T> ModuleParameters for Box<T>
where
    T: ModuleParameters + ?Sized,
{
    fn parameters(&self) -> ModuleParamRef<'_> {
        self.as_ref().parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        self.as_mut().parameters_mut()
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.as_ref().trainable_parameters()
    }
}

impl<T> ModuleParameters for Vec<T>
where
    T: ModuleParameters,
{
    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter().for_each(|module| {
            let module_parameters = module.parameters();
            parameters.entries.extend(module_parameters.entries);
        });
        parameters
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter_mut().for_each(|module| {
            let module_parameters = module.parameters_mut();
            parameters.entries.extend(module_parameters.entries);
        });
        parameters
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter().for_each(|module| {
            let module_parameters = module.trainable_parameters();
            parameters.entries.extend(module_parameters.entries);
        });
        parameters
    }
}
