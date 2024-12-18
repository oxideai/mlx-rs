use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    nested::{NestedHashMap, NestedValue},
    Array,
};

/// Type alias for owned module parameters.
pub type ModuleParam = NestedHashMap<Rc<str>, Array>;

/// Type alias for borrowed module parameters.
pub type ModuleParamRef<'a> = NestedHashMap<Rc<str>, &'a RefCell<Array>>;

/// Type alias for flattened module parameters.
pub type FlattenedModuleParam = HashMap<Rc<str>, Array>;

/// Type alias for borrowed flattened module parameters.
pub type FlattenedModuleParamRef<'a> = HashMap<Rc<str>, &'a RefCell<Array>>;

/// Trait for a neural network module.
pub trait Module<Args>: ModuleParameters + std::fmt::Debug {
    /// Error type for the module.
    type Error: std::error::Error;

    /// Output type of the module.
    type Output;

    /// Forward pass of the module.
    fn forward(&self, x: Args) -> Result<Self::Output, Self::Error>;

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
pub trait UnaryModule
where
    for<'a> Self: Module<&'a Array, Output = Array>,
{
}

impl<M> UnaryModule for M where for<'a> M: Module<&'a Array, Output = Array> {}

/// Trait for accessing and updating module parameters.
pub trait ModuleParameters {
    /// Get references to the module parameters.
    fn parameters(&self) -> ModuleParamRef<'_>;

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
pub fn update_flattened_parameters<M, I>(module: &M, flattened_parameters: I)
where
    M: ModuleParameters + ?Sized,
    I: IntoIterator<Item = (Rc<str>, Array)>,
{
    let flattened_self_parameters = module.parameters().flatten();

    for (key, value) in flattened_parameters {
        if let Some(self_value) = flattened_self_parameters.get(&key) {
            *self_value.borrow_mut() = value;
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
