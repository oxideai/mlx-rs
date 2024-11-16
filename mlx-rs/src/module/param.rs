use std::{
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

    /// Check if the parameter is frozen.
    fn is_frozen(&self) -> bool;

    /// Get the parameter as a nested value.
    fn as_nested_value<'a>(&self) -> NestedValue<&'a str, &Array>;

    /// Get the parameter as a mutable nested value.
    fn as_nested_value_mut<'a>(&mut self) -> NestedValue<&'a str, &mut Array>;

    /// Get the parameter as a nested value if it is trainable.
    fn as_trainable_nested_value<'a>(&self) -> Option<NestedValue<&'a str, &Array>>;
}

/// A simple wrapper for a module parameter.
#[derive(Debug, Clone)]
pub struct Param<T> {
    /// The value of the parameter.
    pub value: T,

    /// Whether the parameter is frozen.
    ///
    /// This is no longer public because it should be accessed through the `Parameter` trait.
    is_frozen: bool,
}

impl<T> Param<T> {
    /// Create a new `Param`
    pub fn new(value: T) -> Self {
        Self {
            value,
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
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for Param<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T> AsRef<T> for Param<T> {
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T> AsMut<T> for Param<T> {
    fn as_mut(&mut self) -> &mut T {
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

    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn as_nested_value<'a>(&self) -> NestedValue<&'a str, &Array> {
        NestedValue::Value(&self.value)
    }

    fn as_nested_value_mut<'a>(&mut self) -> NestedValue<&'a str, &mut Array> {
        NestedValue::Value(&mut self.value)
    }

    fn as_trainable_nested_value<'a>(&self) -> Option<NestedValue<&'a str, &Array>> {
        match self.is_frozen {
            true => None,
            false => Some(NestedValue::Value(&self.value)),
        }
    }
}

impl Parameter for Param<Option<Array>> {
    fn freeze(&mut self, _recursive: bool) {
        self.is_frozen = true;
    }

    fn unfreeze(&mut self, _recursive: bool) {
        self.is_frozen = false;
    }

    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn as_nested_value<'a>(&self) -> NestedValue<&'a str, &Array> {
        match &self.value {
            Some(array) => NestedValue::Value(array),
            // An empty map entry will be ignored during flattening
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }

    fn as_nested_value_mut<'a>(&mut self) -> NestedValue<&'a str, &mut Array> {
        match &mut self.value {
            Some(array) => NestedValue::Value(array),
            // An empty map entry will be ignored during flattening
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }

    fn as_trainable_nested_value<'a>(&self) -> Option<NestedValue<&'a str, &Array>> {
        match self.is_frozen {
            true => None,
            false => self.value.as_ref().map(NestedValue::Value),
        }
    }
}

impl<T> Parameter for Param<T>
where
    T: ModuleParameters,
{
    fn freeze(&mut self, recursive: bool) {
        self.value.freeze_parameters(recursive);
        self.is_frozen = true;
    }

    fn unfreeze(&mut self, recursive: bool) {
        self.value.unfreeze_parameters(recursive);
        self.is_frozen = false;
    }

    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn as_nested_value<'a>(&self) -> NestedValue<&'a str, &Array> {
        self.parameters().into()
    }

    fn as_nested_value_mut<'a>(&mut self) -> NestedValue<&'a str, &mut Array> {
        self.parameters_mut().into()
    }

    fn as_trainable_nested_value<'a>(&self) -> Option<NestedValue<&'a str, &Array>> {
        match self.is_frozen {
            true => None,
            false => Some(self.trainable_parameters().into()),
        }
    }
}
