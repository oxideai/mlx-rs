//! Utility types and functions.

/// Helper type to represent either a single value or a pair of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingleOrPair<T = i32> {
    /// Single value.
    Single(T),

    /// Pair of values.
    Pair(T, T),
}

impl<T> From<T> for SingleOrPair<T> {
    fn from(value: T) -> Self {
        SingleOrPair::Single(value)
    }
}

impl<T> From<(T, T)> for SingleOrPair<T> {
    fn from(value: (T, T)) -> Self {
        SingleOrPair::Pair(value.0, value.1)
    }
}

impl<T: Clone> From<SingleOrPair<T>> for (T, T) {
    fn from(value: SingleOrPair<T>) -> Self {
        match value {
            SingleOrPair::Single(v) => (v.clone(), v),
            SingleOrPair::Pair(v1, v2) => (v1, v2),
        }
    }
}

/// Helper type to represent either a single value or a triple of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingleOrTriple<T = i32> {
    /// Single value.
    Single(T),

    /// Triple of values.
    Triple(T, T, T),
}

impl<T> From<T> for SingleOrTriple<T> {
    fn from(value: T) -> Self {
        SingleOrTriple::Single(value)
    }
}

impl<T> From<(T, T, T)> for SingleOrTriple<T> {
    fn from(value: (T, T, T)) -> Self {
        SingleOrTriple::Triple(value.0, value.1, value.2)
    }
}

impl<T: Clone> From<SingleOrTriple<T>> for (T, T, T) {
    fn from(value: SingleOrTriple<T>) -> Self {
        match value {
            SingleOrTriple::Single(v) => (v.clone(), v.clone(), v),
            SingleOrTriple::Triple(v1, v2, v3) => (v1, v2, v3),
        }
    }
}
