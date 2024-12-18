use std::{
    cell::RefCell,
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use crate::{nested::NestedValue, Array};

use super::ModuleParameters;

/// Trait for a module parameter.
pub trait Parameter {
    /// Freeze the parameter.
    fn freeze(&mut self, recursive: bool);

    /// Unfreeze the parameter.
    fn unfreeze(&mut self, recursive: bool);

    /// Check if the parameter is frozen. Returns `None` if the parameter is a module that has no
    /// parameters.
    fn is_frozen(&self) -> Option<bool>;

    /// Get the parameter as a nested value.
    fn as_nested_value<'a>(&self) -> NestedValue<&'a str, &RefCell<Array>>;

    /// Get the parameter as a nested value if it is trainable.
    fn as_trainable_nested_value<'a>(&self) -> Option<NestedValue<&'a str, &RefCell<Array>>>;
}

/// A simple wrapper for a module parameter.
#[derive(Debug, Clone)]
pub struct Param<T> {
    /// The value of the parameter.
    pub value: RefCell<T>,

    /// Whether the parameter is frozen.
    ///
    /// This is no longer public because it should be accessed through the `Parameter` trait.
    is_frozen: bool,
}

impl<T> Param<T> {
    /// Create a new `Param`
    pub fn new(value: T) -> Self {
        Self {
            value: RefCell::new(value),
            is_frozen: false,
        }
    }
}

impl<T> From<T> for Param<T> {
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}

impl<T> Deref for Param<T> {
    type Target = RefCell<T>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for Param<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl Parameter for Param<Array> {
    fn freeze(&mut self, _recursive: bool) {
        self.is_frozen = true;
    }

    fn unfreeze(&mut self, _recursive: bool) {
        self.is_frozen = false;
    }

    fn is_frozen(&self) -> Option<bool> {
        Some(self.is_frozen)
    }

    fn as_nested_value<'a>(&self) -> NestedValue<&'a str, &RefCell<Array>> {
        NestedValue::Value(&self.value)
    }

    fn as_trainable_nested_value<'a>(&self) -> Option<NestedValue<&'a str, &RefCell<Array>>> {
        match self.is_frozen {
            true => None,
            false => Some(NestedValue::Value(&self.value)),
        }
    }
}

impl Parameter for Option<Param<Array>> {
    fn freeze(&mut self, _recursive: bool) {
        if let Some(param) = self.as_mut() {
            param.freeze(_recursive)
        }
    }

    fn unfreeze(&mut self, _recursive: bool) {
        if let Some(param) = self.as_mut() {
            param.unfreeze(_recursive)
        }
    }

    fn is_frozen(&self) -> Option<bool> {
        self.as_ref().map(|param| param.is_frozen).or(Some(true))
    }

    fn as_nested_value<'a>(&self) -> NestedValue<&'a str, &RefCell<Array>> {
        self.as_ref().map_or_else(
            || NestedValue::Map(HashMap::with_capacity(0)),
            |param| NestedValue::Value(&param.value),
        )
    }

    fn as_trainable_nested_value<'a>(&self) -> Option<NestedValue<&'a str, &RefCell<Array>>> {
        self.as_ref().and_then(|param| {
            if param.is_frozen {
                None
            } else {
                Some(NestedValue::Value(&param.value))
            }
        })
    }
}

impl<T> Parameter for T
where
    T: ModuleParameters,
{
    fn freeze(&mut self, recursive: bool) {
        self.freeze_parameters(recursive);
    }

    fn unfreeze(&mut self, recursive: bool) {
        self.unfreeze_parameters(recursive);
    }

    fn is_frozen(&self) -> Option<bool> {
        self.all_frozen()
    }

    fn as_nested_value<'a>(&self) -> NestedValue<&'a str, &RefCell<Array>> {
        self.parameters().into()
    }

    fn as_trainable_nested_value<'a>(&self) -> Option<NestedValue<&'a str, &RefCell<Array>>> {
        Some(self.trainable_parameters().into())
    }
}
