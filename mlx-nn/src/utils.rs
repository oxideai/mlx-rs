//! Utility types and functions.

use std::rc::Rc;

use mlx_nn_module::FlattenedModuleParam;
use mlx_rs::Array;

/// A custom type to indicate whether a `Module` should include a bias or not.
/// Default to `Yes`.
#[derive(Debug, Clone, Copy, Default)]
pub enum WithBias {
    /// Include a bias in the module.
    #[default]
    Yes,

    /// Do not include a bias in the module.
    No,
}

impl From<bool> for WithBias {
    fn from(value: bool) -> Self {
        if value {
            Self::Yes
        } else {
            Self::No
        }
    }
}

impl From<WithBias> for bool {
    fn from(value: WithBias) -> Self {
        match value {
            WithBias::Yes => true,
            WithBias::No => false,
        }
    }
}

impl WithBias {
    /// Transforms [`WithBias`] into an [`Option`] by applying the given function `f`, mapping
    /// `WithBias::Yes` to `Some(T)` and `WithBias::No` to `None`.
    pub fn map_into_option<F, T>(self, f: F) -> Option<T>
    where
        F: FnOnce() -> T,
    {
        match self {
            WithBias::Yes => Some(f()),
            WithBias::No => None,
        }
    }
}

/// A convenience trait to convert a single value or a pair of values into a pair of values.
pub trait IntOrPair {
    /// Converts the value into a pair of values.
    fn into_pair(self) -> (i32, i32);
}

impl IntOrPair for i32 {
    fn into_pair(self) -> (i32, i32) {
        (self, self)
    }
}

impl IntOrPair for (i32, i32) {
    fn into_pair(self) -> (i32, i32) {
        self
    }
}

/// A convenience trait to convert a single value or a triple of values into a triple of values.
pub trait IntOrTriple {
    /// Converts the value into a triple of values.
    fn into_triple(self) -> (i32, i32, i32);
}

impl IntOrTriple for i32 {
    fn into_triple(self) -> (i32, i32, i32) {
        (self, self, self)
    }
}

impl IntOrTriple for (i32, i32, i32) {
    fn into_triple(self) -> (i32, i32, i32) {
        self
    }
}

pub(crate) fn get_mut_or_insert_with<'a>(
    map: &'a mut FlattenedModuleParam,
    key: &Rc<str>,
    f: impl FnOnce() -> Array,
) -> &'a mut Array {
    if !map.contains_key(key) {
        map.insert(key.clone(), f());
    }

    map.get_mut(key).unwrap()
}
