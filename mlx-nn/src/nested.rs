pub enum NestedItem<V> {
    Array(Vec<V>),
    Nested(Box<Self>),
}