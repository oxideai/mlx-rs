//! Utility types and functions.

/// A convenience trait to convert a single value or a pair of values into a pair of values.
pub trait SingleOrPair<Item> {
    /// Converts the value into a pair of values.
    fn into_pair(self) -> (Item, Item);
}

impl<T> SingleOrPair<T> for T 
where 
    T: Copy
{
    fn into_pair(self) -> (T, T) {
        (self, self)
    }
}

impl<T> SingleOrPair<T> for (T, T) {
    fn into_pair(self) -> (T, T) {
        self
    }
}

/// A convenience trait to convert a single value or a triple of values into a triple of values.
pub trait SingleOrTriple<Item> {
    /// Converts the value into a triple of values.
    fn into_triple(self) -> (Item, Item, Item);
}

impl<T> SingleOrTriple<T> for T 
where 
    T: Copy
{
    fn into_triple(self) -> (T, T, T) {
        (self, self, self)
    }
}

impl<T> SingleOrTriple<T> for (T, T, T) {
    fn into_triple(self) -> (T, T, T) {
        self
    }
}
