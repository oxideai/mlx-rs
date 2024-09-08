use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use mlx_rs::{nested::NestedValue, Array};

pub trait Parameter {
    fn freeze(&mut self);
    fn unfreeze(&mut self);

    fn is_frozen(&self) -> bool;

    fn as_nested_value<K>(&self) -> NestedValue<K, &Array>;

    fn as_nested_value_mut<K>(&mut self) -> NestedValue<K, &mut Array>;
}

pub struct Param<T> {
    pub value: T,
    pub is_frozen: bool,
}

impl<T> Param<T> {
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
    fn freeze(&mut self) {
        self.is_frozen = true;
    }

    fn unfreeze(&mut self) {
        self.is_frozen = false;
    }

    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn as_nested_value<K>(&self) -> NestedValue<K, &Array> {
        NestedValue::Value(&self.value)
    }

    fn as_nested_value_mut<K>(&mut self) -> NestedValue<K, &mut Array> {
        NestedValue::Value(&mut self.value)
    }
}

impl Parameter for Param<Option<Array>> {
    fn freeze(&mut self) {
        self.is_frozen = true;
    }

    fn unfreeze(&mut self) {
        self.is_frozen = false;
    }

    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn as_nested_value<K>(&self) -> NestedValue<K, &Array> {
        match &self.value {
            Some(array) => NestedValue::Value(array),
            // An empty map entry will be ignored during flattening
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }

    fn as_nested_value_mut<K>(&mut self) -> NestedValue<K, &mut Array> {
        match &mut self.value {
            Some(array) => NestedValue::Value(array),
            // An empty map entry will be ignored during flattening
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }
}
