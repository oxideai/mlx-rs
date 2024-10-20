use std::{collections::HashMap, rc::Rc};

use crate::{nested::NestedHashMap, Array};

/// Type alias for owned module parameters.
pub type ModuleParam = NestedHashMap<&'static str, Array>;

/// Type alias for borrowed module parameters.
pub type ModuleParamRef<'a> = NestedHashMap<&'static str, &'a Array>;

/// Type alias for mutably borrowed module parameters.
pub type ModuleParamMut<'a> = NestedHashMap<&'static str, &'a mut Array>;

/// Type alias for flattened module parameters.
pub type FlattenedModuleParam = HashMap<Rc<str>, Array>;

/// Type alias for borrowed flattened module parameters.
pub type FlattenedModuleParamRef<'a> = HashMap<Rc<str>, &'a Array>;

/// Type alias for mutably borrowed flattened module parameters.
pub type FlattenedModuleParamMut<'a> = HashMap<Rc<str>, &'a mut Array>;

/// Trait for a neural network module.
pub trait Module: ModuleParameters {
    /// Error type for the module.
    type Error: std::error::Error;

    /// Forward pass of the module.
    fn forward(&self, x: &Array) -> Result<Array, Self::Error>;

    /// Set whether the module is in training mode.
    ///
    /// Training mode only applies to certain layers. For example, dropout layers applies a random
    /// mask in training mode, but is the identity in evaluation mode. Implementations of nested
    /// modules should propagate the training mode to all child modules.
    fn training_mode(&mut self, mode: bool);
}

/// Trait for accessing and updating module parameters.
pub trait ModuleParameters {
    /// Get references to the module parameters.
    fn parameters(&self) -> ModuleParamRef<'_>;

    /// Get mutable references to the module parameters.
    fn parameters_mut(&mut self) -> ModuleParamMut<'_>;

    /// Get references to the trainable parameters. A parameter is trainable if it is NOT frozen.
    fn trainable_parameters(&self) -> ModuleParamRef<'_>;

    /// Update the module parameters.
    fn update(&mut self, parameters: ModuleParam) {
        let flattened_parameters = parameters.flatten();
        update_flattened_parameters(self, flattened_parameters)
    }

    /// Update the module parameters from a flattened representation.
    fn update_flattened(&mut self, flattened_parameters: FlattenedModuleParam) {
        update_flattened_parameters(self, flattened_parameters)
    }
}

/// Update the module parameters from an iterator of flattened parameters.
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
