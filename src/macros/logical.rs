/// See [`crate::Array::eq`] for details.
#[macro_export]
macro_rules! eq {
    ($a:expr, $b:expr) => {
        $a.eq($b.as_ref())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.eq_device($b.as_ref(), $stream)
    };
}

/// See [`crate::Array::le`] for details.
#[macro_export]
macro_rules! le {
    ($a:expr, $b:expr) => {
        $a.le($b.as_ref())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.le_device($b.as_ref(), $stream)
    };
}

/// See [`crate::Array::ge`] for details.
#[macro_export]
macro_rules! ge {
    ($a:expr, $b:expr) => {
        $a.ge($b.as_ref())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.ge_device($b.as_ref(), $stream)
    };
}

/// See [`crate::Array::ne`] for details.
#[macro_export]
macro_rules! ne {
    ($a:expr, $b:expr) => {
        $a.ne($b.as_ref())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.ne_device($b.as_ref(), $stream)
    };
}

/// See [`crate::Array::lt`] for details.
#[macro_export]
macro_rules! lt {
    ($a:expr, $b:expr) => {
        $a.lt($b.as_ref())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.lt_device($b.as_ref(), $stream)
    };
}

/// See [`crate::Array::gt`] for details.
#[macro_export]
macro_rules! gt {
    ($a:expr, $b:expr) => {
        $a.gt($b.as_ref())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.gt_device($b.as_ref(), $stream)
    };
}

/// See [`crate::Array::logical_and`] for details.
#[macro_export]
macro_rules! logical_and {
    ($a:expr, $b:expr) => {
        $a.logical_and($b.as_ref())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.logical_and_device($b.as_ref(), $stream)
    };
}

/// See [`crate::Array::logical_or`] for details.
#[macro_export]
macro_rules! logical_or {
    ($a:expr, $b:expr) => {
        $a.logical_or($b.as_ref())
    };
    ($a:expr, $b:expr, stream=$stream:expr) => {
        $a.logical_or_device($b.as_ref(), $stream)
    };
}

/// See [`crate::Array::all_close`] for details.
#[macro_export]
macro_rules! all_close {
    ($a:expr, $b:expr) => {
        $a.all_close($b.as_ref(), None, None, None)
    };
    ($a:expr, $b:expr, $rtol:expr) => {
        $a.all_close($b.as_ref(), $rtol, None, None)
    };
    ($a:expr, $b:expr, $rtol:expr, $atol:expr) => {
        $a.all_close($b.as_ref(), $rtol, $atol, None)
    };
    ($a:expr, $b:expr, $rtol:expr, $atol:expr, $equal_nan:expr) => {
        $a.all_close($b.as_ref(), $rtol, $atol, $equal_nan)
    };
    ($a:expr, $b:expr, $rtol:expr, $atol:expr, $equal_nan:expr, stream=$stream:expr) => {
        $a.all_close_device($b.as_ref(), $rtol, $atol, $equal_nan, $stream)
    };
}

/// See [`crate::Array::is_close`] for details.
#[macro_export]
macro_rules! is_close {
    ($a:expr, $b:expr) => {
        $a.is_close($b.as_ref(), None, None, None)
    };
    ($a:expr, $b:expr, $rtol:expr) => {
        $a.is_close($b.as_ref(), $rtol, None, None)
    };
    ($a:expr, $b:expr, $rtol:expr, $atol:expr) => {
        $a.is_close($b.as_ref(), $rtol, $atol, None)
    };
    ($a:expr, $b:expr, $rtol:expr, $atol:expr, $equal_nan:expr) => {
        $a.is_close($b.as_ref(), $rtol, $atol, $equal_nan)
    };
    ($a:expr, $b:expr, $rtol:expr, $atol:expr, $equal_nan:expr, stream=$stream:expr) => {
        $a.is_close_device($b.as_ref(), $rtol, $atol, $equal_nan, $stream)
    };
}

/// See [`crate::Array::array_eq`] for details.
#[macro_export]
macro_rules! array_eq {
    ($a:expr, $b:expr) => {
        $a.array_eq($b.as_ref(), None)
    };
    ($a:expr, $b:expr, $equal_nan:expr) => {
        $a.array_eq($b.as_ref(), $equal_nan)
    };
    ($a:expr, $b:expr, $equal_nan:expr, stream=$stream:expr) => {
        $a.array_eq_device($b.as_ref(), $equal_nan, $stream)
    };
}

/// See [`crate::Array::any`] for details.
#[macro_export]
macro_rules! any {
    ($a:expr) => {
        $a.any(None, None)
    };
    ($a:expr, $axes:expr) => {
        $a.any($axes, None)
    };
    ($a:expr, $axes:expr, $keep_dims:expr) => {
        $a.any($axes, $keep_dims)
    };
    ($a:expr, $axes:expr, $keep_dims:expr, stream=$stream:expr) => {
        $a.any_device($axes, $keep_dims, $stream)
    };
}

/// See [`crate::ops::which`] for details.
#[macro_export]
macro_rules! which {
    ($condition:expr, $a:expr, $b:expr) => {
        $crate::ops::which($condition.as_ref(), $a.as_ref(), $b.as_ref())
    };
    ($condition:expr, $a:expr, $b:expr, stream=$stream:expr) => {
        $crate::ops::which_device($condition.as_ref(), $a.as_ref(), $b.as_ref(), $stream)
    };
}

#[cfg(test)]
mod tests {
    use crate::{Array, StreamOrDevice};

    #[test]
    fn test_eq() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);

        // We are just testing that the macros compile
        let _c = eq!(a, b);
        let _c = eq!(a, b, stream=&stream);
    }

    #[test]
    fn test_le() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);

        // We are just testing that the macros compile
        let _c = le!(a, b);
        let _c = le!(a, b, stream=&stream);
    }

    #[test]
    fn test_ge() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);

        // We are just testing that the macros compile
        let _c = ge!(a, b);
        let _c = ge!(a, b, stream=&stream);
    }

    #[test]
    fn test_ne() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);

        // We are just testing that the macros compile
        let _c = ne!(a, b);
        let _c = ne!(a, b, stream=&stream);
    }

    #[test]
    fn test_lt() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);

        // We are just testing that the macros compile
        let _c = lt!(a, b);
        let _c = lt!(a, b, stream=&stream);
    }

    #[test]
    fn test_gt() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);

        // We are just testing that the macros compile
        let _c = gt!(a, b);
        let _c = gt!(a, b, stream=&stream);
    }

    #[test]
    fn test_logical_and() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, false, true], &[3]);

        // We are just testing that the macros compile
        let _c = logical_and!(a, b);
        let _c = logical_and!(a, b, stream=&stream);
    }

    #[test]
    fn test_logical_or() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, false, true], &[3]);

        // We are just testing that the macros compile
        let _c = logical_or!(a, b);
        let _c = logical_or!(a, b, stream=&stream);
    }

    #[test]
    fn test_all_close() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macros compile
        let _c = all_close!(a, b);
        let _c = all_close!(a, b, 1e-6);
        let _c = all_close!(a, b, 1e-6, 1e-6);
        let _c = all_close!(a, b, 1e-6, 1e-6, true);
        let _c = all_close!(a, b, 1e-6, 1e-6, true, stream=&stream);
    }

    #[test]
    fn test_is_close() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macros compile
        let _c = is_close!(a, b);
        let _c = is_close!(a, b, 1e-6);
        let _c = is_close!(a, b, 1e-6, 1e-6);
        let _c = is_close!(a, b, 1e-6, 1e-6, true);
        let _c = is_close!(a, b, 1e-6, 1e-6, true, stream=&stream);
    }

    #[test]
    fn test_array_eq() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);

        // We are just testing that the macros compile
        let _c = array_eq!(a, b);
        let _c = array_eq!(a, b, true);
        let _c = array_eq!(a, b, true, stream=&stream);
    }

    #[test]
    fn test_any() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[true, false, true], &[3]);

        // We are just testing that the macros compile
        let _c = any!(a);
        let _c = any!(a, &[0][..]);
        let _c = any!(a, &[0][..], true);
        let _c = any!(a, &[0][..], true, stream=&stream);
    }

    #[test]
    fn test_which() {
        let stream = StreamOrDevice::default();
        let condition = Array::from_slice(&[true, false, true], &[3]);
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[4, 5, 6], &[3]);

        // We are just testing that the macros compile
        let _c = which!(condition, a, b);
        let _c = which!(condition, a, b, stream=&stream);
    }
}
