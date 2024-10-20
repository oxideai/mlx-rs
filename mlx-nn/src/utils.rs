//! Utility types and functions.

use std::rc::Rc;

use mlx_rs::module::FlattenedModuleParam;
use mlx_rs::Array;

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
