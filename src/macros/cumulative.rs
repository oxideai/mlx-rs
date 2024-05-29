/// See [`crate::Array::cummax`] for more information.
#[macro_export]
macro_rules! cummax {
    ($a:expr) => {
        $a.cummax(None, None, None)
    };
    ($a:expr, $axis:expr) => {
        $a.cummax($axis, None, None)
    };
    ($a:expr, $axis:expr, $reverse:expr) => {
        $a.cummax($axis, $reverse, None)
    };
    ($a:expr, $axis:expr, $reverse:expr, $inclusive:expr) => {
        $a.cummax($axis, $reverse, $inclusive)
    };
    ($a:expr, $axis:expr, $reverse:expr, $inclusive:expr, stream=$stream:expr) => {
        $a.cummax_device($axis, $reverse, $inclusive, $stream)
    };
}

/// See [`crate::Array::cummin`] for more information.
#[macro_export]
macro_rules! cummin {
    ($a:expr) => {
        $a.cummin(None, None, None)
    };
    ($a:expr, $axis:expr) => {
        $a.cummin($axis, None, None)
    };
    ($a:expr, $axis:expr, $reverse:expr) => {
        $a.cummin($axis, $reverse, None)
    };
    ($a:expr, $axis:expr, $reverse:expr, $inclusive:expr) => {
        $a.cummin($axis, $reverse, $inclusive)
    };
    ($a:expr, $axis:expr, $reverse:expr, $inclusive:expr, stream=$stream:expr) => {
        $a.cummin_device($axis, $reverse, $inclusive, $stream)
    };
}

/// See [`crate::Array::cumprod`] for more information.
#[macro_export]
macro_rules! cumprod {
    ($a:expr) => {
        $a.cumprod(None, None, None)
    };
    ($a:expr, $axis:expr) => {
        $a.cumprod($axis, None, None)
    };
    ($a:expr, $axis:expr, $reverse:expr) => {
        $a.cumprod($axis, $reverse, None)
    };
    ($a:expr, $axis:expr, $reverse:expr, $inclusive:expr) => {
        $a.cumprod($axis, $reverse, $inclusive)
    };
    ($a:expr, $axis:expr, $reverse:expr, $inclusive:expr, stream=$stream:expr) => {
        $a.cumprod_device($axis, $reverse, $inclusive, $stream)
    };
}

#[macro_export]
macro_rules! cumsum {
    ($a:expr) => {
        $a.cumsum(None, None, None)
    };
    ($a:expr, $axis:expr) => {
        $a.cumsum($axis, None, None)
    };
    ($a:expr, $axis:expr, $reverse:expr) => {
        $a.cumsum($axis, $reverse, None)
    };
    ($a:expr, $axis:expr, $reverse:expr, $inclusive:expr) => {
        $a.cumsum($axis, $reverse, $inclusive)
    };
    ($a:expr, $axis:expr, $reverse:expr, $inclusive:expr, stream=$stream:expr) => {
        $a.cumsum_device($axis, $reverse, $inclusive, $stream)
    };
}

#[cfg(test)]
mod tests {
    use crate::{Array, StreamOrDevice};

    #[test]
    fn test_cummax() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        // We are just testing that the macro compiles
        let _b = cummax!(a);
        let _b = cummax!(a, 0);
        let _b = cummax!(a, 0, true);
        let _b = cummax!(a, 0, true, true);
        let _b = cummax!(a, 0, true, true, stream=stream);
    }

    #[test]
    fn test_cummin() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        // We are just testing that the macro compiles
        let _b = cummin!(a);
        let _b = cummin!(a, 0);
        let _b = cummin!(a, 0, true);
        let _b = cummin!(a, 0, true, true);
        let _b = cummin!(a, 0, true, true, stream=stream);
    }

    #[test]
    fn test_cumprod() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        // We are just testing that the macro compiles
        let _b = cumprod!(a);
        let _b = cumprod!(a, 0);
        let _b = cumprod!(a, 0, true);
        let _b = cumprod!(a, 0, true, true);
        let _b = cumprod!(a, 0, true, true, stream=stream);
    }

    #[test]
    fn test_cumsum() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        // We are just testing that the macro compiles
        let _b = cumsum!(a);
        let _b = cumsum!(a, 0);
        let _b = cumsum!(a, 0, true);
        let _b = cumsum!(a, 0, true, true);
        let _b = cumsum!(a, 0, true, true, stream=stream);
    }
}