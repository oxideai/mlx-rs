/// See [`crate::Array::all`] for more information.
#[macro_export]
macro_rules! all {
    ($a:expr) => {
        $a.all(None, None)
    };
    ($a:expr, $axes:expr) => {
        $a.all($axes, None)
    };
    ($a:expr, $axes:expr, $keepdims:expr) => {
        $a.all($axes, $keepdims)
    };
    ($a:expr, $axes:expr, $keepdims:expr, $stream:expr) => {
        $a.all_device($axes, $keepdims, $stream)
    };
}

/// See [`crate::Array::prod`] for more information.
#[macro_export]
macro_rules! prod {
    ($a:expr) => {
        $a.prod(None, None)
    };
    ($a:expr, $axes:expr) => {
        $a.prod($axes, None)
    };
    ($a:expr, $axes:expr, $keepdims:expr) => {
        $a.prod($axes, $keepdims)
    };
    ($a:expr, $axes:expr, $keepdims:expr, $stream:expr) => {
        $a.prod_device($axes, $keepdims, $stream)
    };
}

/// See [`crate::Array::max`] for more information.
#[macro_export]
macro_rules! max {
    ($a:expr) => {
        $a.max(None, None)
    };
    ($a:expr, $axes:expr) => {
        $a.max($axes, None)
    };
    ($a:expr, $axes:expr, $keepdims:expr) => {
        $a.max($axes, $keepdims)
    };
    ($a:expr, $axes:expr, $keepdims:expr, $stream:expr) => {
        $a.max_device($axes, $keepdims, $stream)
    };
}

/// See [`crate::Array::sum`] for more information.
#[macro_export]
macro_rules! sum {
    ($a:expr) => {
        $a.sum(None, None)
    };
    ($a:expr, $axes:expr) => {
        $a.sum($axes, None)
    };
    ($a:expr, $axes:expr, $keepdims:expr) => {
        $a.sum($axes, $keepdims)
    };
    ($a:expr, $axes:expr, $keepdims:expr, $stream:expr) => {
        $a.sum_device($axes, $keepdims, $stream)
    };
}

/// See [`crate::Array::mean`] for more information.
#[macro_export]
macro_rules! mean {
    ($a:expr) => {
        $a.mean(None, None)
    };
    ($a:expr, $axes:expr) => {
        $a.mean($axes, None)
    };
    ($a:expr, $axes:expr, $keepdims:expr) => {
        $a.mean($axes, $keepdims)
    };
    ($a:expr, $axes:expr, $keepdims:expr, $stream:expr) => {
        $a.mean_device($axes, $keepdims, $stream)
    };
}

/// See [`crate::Array::min`] for more information.
#[macro_export]
macro_rules! min {
    ($a:expr) => {
        $a.min(None, None)
    };
    ($a:expr, $axes:expr) => {
        $a.min($axes, None)
    };
    ($a:expr, $axes:expr, $keepdims:expr) => {
        $a.min($axes, $keepdims)
    };
    ($a:expr, $axes:expr, $keepdims:expr, $stream:expr) => {
        $a.min_device($axes, $keepdims, $stream)
    };
}

/// See [`crate::Array::variance`] for more information.
#[macro_export]
macro_rules! variance {
    ($a:expr) => {
        $a.variance(None, None, None)
    };
    ($a:expr, $axes:expr) => {
        $a.variance($axes, None, None)
    };
    ($a:expr, $axes:expr, $keepdims:expr) => {
        $a.variance($axes, $keepdims, None)
    };
    ($a:expr, $axes:expr, $keepdims:expr, $ddof:expr) => {
        $a.variance($axes, $keepdims, $ddof)
    };
    ($a:expr, $axes:expr, $keepdims:expr, $ddof:expr, $stream:expr) => {
        $a.variance_device($axes, $keepdims, $ddof, $stream)
    };
}

/// See [`crate::Array::log_sum_exp`] for more information.
#[macro_export]
macro_rules! log_sum_exp {
    ($a:expr) => {
        $a.log_sum_exp(None, None)
    };
    ($a:expr, $axes:expr) => {
        $a.log_sum_exp($axes, None)
    };
    ($a:expr, $axes:expr, $keepdims:expr) => {
        $a.log_sum_exp($axes, $keepdims)
    };
    ($a:expr, $axes:expr, $keepdims:expr, $stream:expr) => {
        $a.log_sum_exp_device($axes, $keepdims, $stream)
    };
}

#[cfg(test)]
mod tests {
    use crate::{Array, StreamOrDevice};

    #[test]
    fn test_all() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[true, false, true, false], &[2, 2]);

        // We are just testing that the macros compile
        let _result = all!(&a);
        let _result = all!(a, &[0, 1][..]);
        let _result = all!(a, &[1][..], false);
        let _result = all!(a, &[0][..], true, &stream);
    }

    #[test]
    fn test_prod() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // We are just testing that the macros compile
        let _result = prod!(&a);
        let _result = prod!(a, &[0, 1][..]);
        let _result = prod!(a, &[1][..], false);
        let _result = prod!(a, &[0][..], true, &stream);
    }

    #[test]
    fn test_max() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // We are just testing that the macros compile
        let _result = max!(&a);
        let _result = max!(a, &[0, 1][..]);
        let _result = max!(a, &[1][..], false);
        let _result = max!(a, &[0][..], true, &stream);
    }

    #[test]
    fn test_sum() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // We are just testing that the macros compile
        let _result = sum!(&a);
        let _result = sum!(a, &[0, 1][..]);
        let _result = sum!(a, &[1][..], false);
        let _result = sum!(a, &[0][..], true, &stream);
    }

    #[test]
    fn test_mean() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // We are just testing that the macros compile
        let _result = mean!(&a);
        let _result = mean!(a, &[0, 1][..]);
        let _result = mean!(a, &[1][..], false);
        let _result = mean!(a, &[0][..], true, &stream);
    }

    #[test]
    fn test_min() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // We are just testing that the macros compile
        let _result = min!(&a);
        let _result = min!(a, &[0, 1][..]);
        let _result = min!(a, &[1][..], false);
        let _result = min!(a, &[0][..], true, &stream);
    }

    #[test]
    fn test_variance() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // We are just testing that the macros compile
        let _result = variance!(&a);
        let _result = variance!(a, &[0, 1][..]);
        let _result = variance!(a, &[1][..], false);
        let _result = variance!(a, &[0][..], true, 3);
        let _result = variance!(a, &[0][..], true, 3, &stream);
    }

    #[test]
    fn test_log_sum_exp() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // We are just testing that the macros compile
        let _result = log_sum_exp!(&a);
        let _result = log_sum_exp!(a, &[0, 1][..]);
        let _result = log_sum_exp!(a, &[1][..], false);
        let _result = log_sum_exp!(a, &[0][..], true, &stream);
    }
}
