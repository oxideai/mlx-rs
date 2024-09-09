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
