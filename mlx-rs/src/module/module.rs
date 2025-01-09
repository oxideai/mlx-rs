use std::{collections::HashMap, rc::Rc};

use crate::{
    error::Exception, nested::{NestedHashMap, NestedValue}, Array
};

/// Type alias for owned module parameters.
pub type ModuleParam = NestedHashMap<Rc<str>, Array>;

/// Type alias for borrowed module parameters.
pub type ModuleParamRef<'a> = NestedHashMap<Rc<str>, &'a Array>;

/// Type alias for mutably borrowed module parameters.
pub type ModuleParamMut<'a> = NestedHashMap<Rc<str>, &'a mut Array>;

/// Type alias for flattened module parameters.
pub type FlattenedModuleParam = HashMap<Rc<str>, Array>;

/// Type alias for borrowed flattened module parameters.
pub type FlattenedModuleParamRef<'a> = HashMap<Rc<str>, &'a Array>;

/// Type alias for mutably borrowed flattened module parameters.
pub type FlattenedModuleParamMut<'a> = HashMap<Rc<str>, &'a mut Array>;

/// Trait for a neural network module.
pub trait Module<'a>: ModuleParameters + std::fmt::Debug {
    /// Input type of the module.
    type Input;

    /// Output type of the module.
    type Output;

    /// Error type for the module.
    type Error: std::error::Error;

    /// Forward pass of the module.
    fn forward(&mut self, x: impl Into<Self::Input>) -> Result<Self::Output, Self::Error>;

    /// Set whether the module is in training mode.
    ///
    /// Training mode only applies to certain layers. For example, dropout layers applies a random
    /// mask in training mode, but is the identity in evaluation mode. Implementations of nested
    /// modules should propagate the training mode to all child modules.
    fn training_mode(&mut self, mode: bool);
}

/// Marker trait for a unary neural network module.
///
/// This trait should not be implemented directly. Instead, implement [`Module`] with `Args` as a
/// reference to the input.
pub trait UnaryModule<Err = Exception>: ModuleParameters {
    /// Forward pass of the unary module. This is NOT intended to be called directly outs. Use
    /// [`Module::forward`] instead.
    fn forward_unary(&mut self, x: &Array) -> Result<Array, Err>;

    /// Set whether the module is in training mode. This is NOT intended to be called directly.
    /// Use [`Module::training_mode`] instead.
    fn training_mode_unary(&mut self, mode: bool);
}

impl<M, Err> UnaryModule<Err> for M where for<'a> M: Module<'a, Input = &'a Array, Output = Array, Error = Err> {
    fn forward_unary(&mut self, x: &Array) -> Result<Array, Err> {
        Module::forward(self, x)
    }

    fn training_mode_unary(&mut self, mode: bool) {
        Module::training_mode(self, mode);
    }
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

    /// Freeze all parameters in the module.
    fn freeze_parameters(&mut self, recursive: bool);

    /// Unfreeze all parameters in the module.
    fn unfreeze_parameters(&mut self, recursive: bool);

    /// Check if all parameters in the module are frozen. Returns `None` if there are no parameters.
    fn all_frozen(&self) -> Option<bool>;

    /// Check if any parameter in the module is frozen. Returns `None` if there are no parameters.
    fn any_frozen(&self) -> Option<bool>;
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
            **self_value = value;
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

    fn freeze_parameters(&mut self, recursive: bool) {
        self.as_mut().freeze_parameters(recursive);
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.as_mut().unfreeze_parameters(recursive);
    }

    fn all_frozen(&self) -> Option<bool> {
        self.as_ref().all_frozen()
    }

    fn any_frozen(&self) -> Option<bool> {
        self.as_ref().any_frozen()
    }
}

impl<T> ModuleParameters for Vec<T>
where
    T: ModuleParameters,
{
    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter().enumerate().for_each(|(i, module)| {
            let value = module.parameters();
            parameters.insert(Rc::from(i.to_string()), NestedValue::Map(value.entries));
        });
        parameters
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter_mut().enumerate().for_each(|(i, module)| {
            let value = module.parameters_mut();
            parameters.insert(Rc::from(i.to_string()), NestedValue::Map(value.entries));
        });
        parameters
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter().enumerate().for_each(|(i, module)| {
            let value = module.trainable_parameters();
            parameters.insert(Rc::from(i.to_string()), NestedValue::Map(value.entries));
        });
        parameters
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        self.iter_mut().for_each(|module| {
            module.freeze_parameters(recursive);
        });
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.iter_mut().for_each(|module| {
            module.unfreeze_parameters(recursive);
        });
    }

    fn all_frozen(&self) -> Option<bool> {
        let mut result = None;
        for module in self.iter() {
            match module.all_frozen() {
                Some(true) => result = Some(true),
                Some(false) => return Some(false),
                None => {}
            }
        }
        result
    }

    fn any_frozen(&self) -> Option<bool> {
        let mut result = None;
        for module in self.iter() {
            match module.any_frozen() {
                Some(true) => return Some(true),
                Some(false) => result = Some(false),
                None => {}
            }
        }
        result
    }
}
