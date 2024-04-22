use crate::Array;

/// Helper method to get a string representation of an mlx object.
pub(crate) fn mlx_describe(ptr: *mut ::std::os::raw::c_void) -> Option<String> {
    let mlx_description = unsafe { mlx_sys::mlx_tostring(ptr) };
    let c_str = unsafe { mlx_sys::mlx_string_data(mlx_description) };

    let description = if c_str.is_null() {
        None
    } else {
        Some(unsafe {
            std::ffi::CStr::from_ptr(c_str)
                .to_string_lossy()
                .into_owned()
        })
    };

    unsafe { mlx_sys::mlx_free(mlx_description as *mut std::ffi::c_void) };

    description
}

/// Helper method to check if two arrays are broadcastable.
///
/// Uses the same broadcasting rules as numpy.
/// https://numpy.org/doc/1.20/user/theory.broadcasting.html
///
/// "The size of the trailing axes for both arrays in an operation must
/// either be the same size or one of them must be one."
pub(crate) fn is_broadcastable(a: &[i32], b: &[i32]) -> bool {
    a.iter()
        .rev()
        .zip(b.iter().rev())
        .all(|(a, b)| *a == 1 || *b == 1 || a == b)
}

impl Array {
    /// Helper method to check if an array can be reshaped to a given shape.
    pub fn can_reshape_to(&self, shape: &[i32]) -> bool {
        if self.shape() == shape {
            return true;
        }

        let mut size = 1;
        let mut infer_idx: isize = -1;
        for i in 0..shape.len() {
            if shape[i] == -1 {
                if infer_idx >= 0 {
                    return false;
                }

                infer_idx = i as isize;
            } else {
                size *= shape[i];
            }
        }

        if size > 0 {
            let quotient = self.size() / size as usize;
            if infer_idx >= 0 {
                size *= quotient as i32;
            }
        } else if infer_idx >= 0 {
            return false;
        }

        // validate the reshaping is valid
        if self.size() != size as usize {
            return false;
        }

        return true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_broadcastable() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[2.0, 2.0, 2.0], &[3]);
        assert!(is_broadcastable(a.shape(), b.shape()));

        let a = Array::from_slice(
            &[
                0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0,
            ],
            &[4, 3],
        );
        let b = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert!(is_broadcastable(a.shape(), b.shape()));

        let a = Array::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ],
            &[2, 2, 4],
        );
        let b = Array::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        assert!(is_broadcastable(a.shape(), b.shape()));
    }

    #[test]
    fn test_is_broadcastable_scalar() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b: Array = 2.0.into();
        assert!(is_broadcastable(a.shape(), b.shape()));
    }

    #[test]
    fn test_is_broadcastable_empty() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert!(is_broadcastable(&[], a.shape()));
    }

    #[test]
    fn test_not_broadcastable() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[2.0, 2.0, 2.0, 2.0], &[4]);
        assert!(!is_broadcastable(a.shape(), b.shape()));

        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[2.0, 2.0], &[1, 2]);
        assert!(!is_broadcastable(a.shape(), b.shape()));
    }

    #[test]
    fn test_can_reshape_to() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert!(a.can_reshape_to(&[3]));
        assert!(a.can_reshape_to(&[1, 3]));
        assert!(a.can_reshape_to(&[3, 1]));
        assert!(a.can_reshape_to(&[1, 1, 3]));
        assert!(a.can_reshape_to(&[1, 3, 1]));
        assert!(a.can_reshape_to(&[3, 1, 1]));
        assert!(a.can_reshape_to(&[1, 1, 1, 3]));
        assert!(a.can_reshape_to(&[1, 1, 3, 1]));
        assert!(a.can_reshape_to(&[1, 3, 1, 1]));
        assert!(a.can_reshape_to(&[3, 1, 1, 1]));
        assert!(a.can_reshape_to(&[1, 1, 1, 1, 3]));
        assert!(a.can_reshape_to(&[1, 1, 1, 3, 1]));
        assert!(a.can_reshape_to(&[1, 1, 3, 1, 1]));
        assert!(a.can_reshape_to(&[1, 3, 1, 1, 1]));
        assert!(a.can_reshape_to(&[3, 1, 1, 1, 1]));
        assert!(a.can_reshape_to(&[1, 1, 1, 1, 1, 3]));
        assert!(a.can_reshape_to(&[1, 1, 1, 1, 3, 1]));
        assert!(a.can_reshape_to(&[1, 1, 1, 3, 1, 1]));
        assert!(a.can_reshape_to(&[1, 1, 3, 1, 1, 1]));
        assert!(a.can_reshape_to(&[1, 3, 1, 1, 1, 1]));
        assert!(a.can_reshape_to(&[3, 1, 1, 1, 1, 1]));
    }

    #[test]
    fn test_reshape_negative_dim() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert!(a.can_reshape_to(&[1, -1]));
        assert!(a.can_reshape_to(&[-1, 1]));
        assert!(a.can_reshape_to(&[-1]));
        assert!(a.can_reshape_to(&[1, -1, 1]));
        assert!(a.can_reshape_to(&[-1, 1, 1]));

        assert!(!a.can_reshape_to(&[1, -2]));
    }

    #[test]
    fn test_cannot_reshape_to() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert!(!a.can_reshape_to(&[2]));
        assert!(!a.can_reshape_to(&[2, 2]));
        assert!(!a.can_reshape_to(&[2, 2, 2]));
        assert!(!a.can_reshape_to(&[2, 2, 2, 2]));
        assert!(!a.can_reshape_to(&[2, 2, 2, 2, 2]));
        assert!(!a.can_reshape_to(&[2, 2, 2, 2, 2, 2]));
    }
}
