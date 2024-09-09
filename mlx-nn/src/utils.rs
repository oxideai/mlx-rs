/// A custom type to indicate whether a `Module` should include a bias or not.
/// Default to `Yes`.
#[derive(Debug, Clone, Copy, Default)]
pub enum WithBias {
    #[default]
    Yes,
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

// pub struct IntOrPair {
//     pair: (i32, i32)
// }

// impl From<i32> for IntOrPair {
//     fn from(value: i32) -> Self {
//         Self {
//             pair: (value, value)
//         }
//     }
// }

// impl From<(i32, i32)> for IntOrPair {
//     fn from(value: (i32, i32)) -> Self {
//         Self {
//             pair: value
//         }
//     }
// }

// impl From<IntOrPair> for (i32, i32) {
//     fn from(value: IntOrPair) -> Self {
//         value.pair
//     }
// }

pub trait IntOrPair {
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

pub trait IntOrTriple {
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
