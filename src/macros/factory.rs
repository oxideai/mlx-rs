// Place the keyword args first to prioritize them when the same number of args is found.

#[macro_export]
macro_rules! zeros {
    ($shape:expr) => {
        $crate::Array::zeros::<f32>($shape)
    };
    ($shape:expr, stream=$stream:expr) => {
        $crate::Array::zeros_device::<f32>($shape, $stream)
    };
    ($shape:expr, dtype=$dtype:ty) => {
        $crate::Array::zeros::<$dtype>($shape)
    };
    ($shape:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::zeros_device::<$dtype>($shape, $stream)
    };
}

#[macro_export]
macro_rules! ones {
    ($shape:expr) => {
        $crate::Array::ones::<f32>($shape)
    };
    ($shape:expr, stream=$stream:expr) => {
        $crate::Array::ones_device::<f32>($shape, $stream)
    };
    ($shape:expr, dtype=$dtype:ty) => {
        $crate::Array::ones::<$dtype>($shape)
    };
    ($shape:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::ones_device::<$dtype>($shape, $stream)
    };
}

#[macro_export]
macro_rules! eye {
    ($n:expr) => {
        $crate::Array::eye::<f32>($n, None, None)
    };
    ($n:expr, stream=$stream:expr) => {
        $crate::Array::eye_device::<f32>($n, None, None, $stream)
    };
    ($n:expr, dtype=$dtype:ty) => {
        $crate::Array::eye::<$dtype>($n, None, None)
    };
    ($n:expr, $m:expr) => {
        $crate::Array::eye::<f32>($n, $m, None)
    };
    ($n:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::eye_device::<$dtype>($n, None, None, $stream)
    };
    ($n:expr, $m:expr, stream=$stream:expr) => {
        $crate::Array::eye_device::<f32>($n, $m, None, $stream)
    };
    ($n:expr, $m:expr, dtype=$dtype:ty) => {
        $crate::Array::eye::<$dtype>($n, $m, None)
    };
    ($n:expr, $m:expr, $k:expr) => {
        $crate::Array::eye::<f32>($n, $m, $k)
    };
    ($n:expr, $m:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::eye_device::<$dtype>($n, $m, None, $stream)
    };
    ($n:expr, $m:expr, $k:expr, stream=$stream:expr) => {
        $crate::Array::eye_device::<f32>($n, $m, $k, $stream)
    };
    ($n:expr, $m:expr, $k:expr, dtype=$dtype:ty) => {
        $crate::Array::eye::<$dtype>($n, $m, $k)
    };
    ($n:expr, $m:expr, $k:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::eye_device::<$dtype>($n, $m, $k, $stream)
    };
}

#[macro_export]
macro_rules! full {
    ($shape:expr, $values:expr) => {
        $crate::Array::full::<f32>($shape, $values)
    };
    ($shape:expr, $values:expr, stream=$stream:expr) => {
        $crate::Array::full_device::<f32>($shape, $values, $stream)
    };
    ($shape:expr, $values:expr, dtype=$dtype:ty) => {
        $crate::Array::full::<$dtype>($shape, $values)
    };
    ($shape:expr, $values:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::full_device::<$dtype>($shape, $values, $stream)
    };
}

#[macro_export]
macro_rules! identity {
    ($n:expr) => {
        $crate::Array::identity::<f32>($n)
    };
    ($n:expr, stream=$stream:expr) => {
        $crate::Array::identity_device::<f32>($n, $stream)
    };
    ($n:expr, dtype=$dtype:ty) => {
        $crate::Array::identity::<$dtype>($n)
    };
    ($n:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::identity_device::<$dtype>($n, $stream)
    };
}

#[macro_export]
macro_rules! linspace {
    ($start:expr, $stop:expr) => {
        $crate::Array::linspace::<f32, _>($start, $stop, None)
    };
    ($start:expr, $stop:expr, stream=$stream:expr) => {
        $crate::Array::linspace_device::<f32, _>($start, $stop, None, $stream)
    };
    ($start:expr, $stop:expr, dtype=$dtype:ty) => {
        $crate::Array::linspace::<$dtype, _>($start, $stop, None)
    };
    ($start:expr, $stop:expr, $count:expr) => {
        $crate::Array::linspace::<f32, _>($start, $stop, $count)
    };
    ($start:expr, $stop:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::linspace_device::<$dtype, _>($start, $stop, None, $stream)
    };
    ($start:expr, $stop:expr, $count:expr, stream=$stream:expr) => {
        $crate::Array::linspace_device::<f32, _>($start, $stop, $count, $stream)
    };
    ($start:expr, $stop:expr, $count:expr, dtype=$dtype:ty) => {
        $crate::Array::linspace::<$dtype, _>($start, $stop, $count)
    };
    ($start:expr, $stop:expr, $count:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::linspace_device::<$dtype, _>($start, $stop, $count, $stream)
    };
}

#[macro_export]
macro_rules! repeat {
    ($a:expr, $count:expr, $axis:expr) => {
        $crate::Array::repeat::<f32>($a, $count, $axis)
    };
    ($a:expr, $count:expr, $axis:expr, stream=$stream:expr) => {
        $crate::Array::repeat_device::<f32>($a, $count, $axis, $stream)
    };
    ($a:expr, $count:expr, $axis:expr, dtype=$dtype:ty) => {
        $crate::Array::repeat::<$dtype>($a, $count, $axis)
    };
    ($a:expr, $count:expr, $axis:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::repeat_device::<$dtype>($a, $count, $axis, $stream)
    };
}

#[macro_export]
macro_rules! repeat_all {
    ($a:expr, $count:expr) => {
        $crate::Array::repeat_all::<f32>($a, $count)
    };
    ($a:expr, $count:expr, stream=$stream:expr) => {
        $crate::Array::repeat_all_device::<f32>($a, $count, $stream)
    };
    ($a:expr, $count:expr, dtype=$dtype:ty) => {
        $crate::Array::repeat_all::<$dtype>($a, $count)
    };
    ($a:expr, $count:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::repeat_all_device::<$dtype>($a, $count, $stream)
    };
}

#[macro_export]
macro_rules! tri {
    ($n:expr) => {
        $crate::Array::tri::<f32>($n, None, None)
    };
    ($n:expr, stream=$stream:expr) => {
        $crate::Array::tri_device::<f32>($n, None, None, $stream)
    };
    ($n:expr, dtype=$dtype:ty) => {
        $crate::Array::tri::<$dtype>($n, None, None)
    };
    ($n:expr, $m:expr) => {
        $crate::Array::tri::<f32>($n, $m, None)
    };
    ($n:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::tri_device::<$dtype>($n, None, None, $stream)
    };
    ($n:expr, $m:expr, stream=$stream:expr) => {
        $crate::Array::tri_device::<f32>($n, $m, None, $stream)
    };
    ($n:expr, $m:expr, dtype=$dtype:ty) => {
        $crate::Array::tri::<$dtype>($n, $m, None)
    };
    ($n:expr, $m:expr, $k:expr) => {
        $crate::Array::tri::<f32>($n, $m, $k)
    };
    ($n:expr, $m:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::tri_device::<$dtype>($n, $m, None, $stream)
    };
    ($n:expr, $m:expr, $k:expr, stream=$stream:expr) => {
        $crate::Array::tri_device::<f32>($n, $m, $k, $stream)
    };
    ($n:expr, $m:expr, $k:expr, dtype=$dtype:ty) => {
        $crate::Array::tri::<$dtype>($n, $m, $k)
    };
    ($n:expr, $m:expr, $k:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $crate::Array::tri_device::<$dtype>($n, $m, $k, $stream)
    };
}

#[cfg(test)]
mod tests {
    use crate::{Array, StreamOrDevice};

    #[test]
    fn test_zeros() {
        let stream = StreamOrDevice::default();

        // We are just testing that the macros compile
        let _a = zeros!(&[2, 3]);
        let _a = zeros!(&[2, 3], stream = &stream);
        let _a = zeros!(&[2, 3], dtype = f32);
        let _a = zeros!(&[2, 3], dtype = f32, stream = &stream);
    }

    #[test]
    fn test_ones() {
        let stream = StreamOrDevice::default();

        // We are just testing that the macros compile
        let _a = ones!(&[2, 3]);
        let _a = ones!(&[2, 3], stream = &stream);
        let _a = ones!(&[2, 3], dtype = f32);
        let _a = ones!(&[2, 3], dtype = f32, stream = &stream);
    }

    #[test]
    fn test_eye() {
        let stream = StreamOrDevice::default();

        // We are just testing that the macros compile
        let _a = eye!(3);
        let _a = eye!(3, stream = &stream);
        let _a = eye!(3, dtype = f32);
        let _a = eye!(3, 4);
        let _a = eye!(3, dtype = f32, stream = &stream);
        let _a = eye!(3, 4, stream = &stream);
        let _a = eye!(3, 4, dtype = f32);
        let _a = eye!(3, 4, 1);
        let _a = eye!(3, 4, dtype = f32, stream = &stream);
        let _a = eye!(3, 4, 1, stream = &stream);
        let _a = eye!(3, 4, 1, dtype = f32);
        let _a = eye!(3, 4, 1, dtype = f32, stream = &stream);
    }

    #[test]
    fn test_full() {
        let stream = StreamOrDevice::default();

        // We are just testing that the macros compile
        let _a = full!(&[2, 3], Array::from_int(5));
        let _a = full!(&[2, 3], Array::from_int(5), stream = &stream);
        let _a = full!(&[2, 3], Array::from_int(5), dtype = f32);
        let _a = full!(&[2, 3], Array::from_int(5), dtype = f32, stream = &stream);
    }

    #[test]
    fn test_identity() {
        let stream = StreamOrDevice::default();

        // We are just testing that the macros compile
        let _a = identity!(3);
        let _a = identity!(3, stream = &stream);
        let _a = identity!(3, dtype = f32);
        let _a = identity!(3, dtype = f32, stream = &stream);
    }

    #[test]
    fn test_linspace() {
        let stream = StreamOrDevice::default();

        // We are just testing that the macros compile
        let _a = linspace!(0.0, 1.0);
        let _a = linspace!(0.0, 1.0, stream = &stream);
        let _a = linspace!(0.0, 1.0, dtype = f32);
        let _a = linspace!(0.0, 1.0, 10);
        let _a = linspace!(0.0, 1.0, dtype = f32, stream = &stream);
        let _a = linspace!(0.0, 1.0, 10, stream = &stream);
        let _a = linspace!(0.0, 1.0, 10, dtype = f32);
        let _a = linspace!(0.0, 1.0, 10, dtype = f32, stream = &stream);
    }

    #[test]
    fn test_repeat() {
        let stream = StreamOrDevice::default();

        // We are just testing that the macros compile
        let a = Array::from_int(5);
        let _a = repeat!(a, 3, 0);

        let a = Array::from_int(5);
        let _a = repeat!(a, 3, 0, stream = &stream);

        let a = Array::from_int(5);
        let _a = repeat!(a, 3, 0, dtype = f32);

        let a = Array::from_int(5);
        let _a = repeat!(a, 3, 0, dtype = f32, stream = &stream);
    }

    #[test]
    fn test_repeat_all() {
        let stream = StreamOrDevice::default();

        // We are just testing that the macros compile
        let a = Array::from_int(5);
        let _a = repeat_all!(a, 3);

        let a = Array::from_int(5);
        let _a = repeat_all!(a, 3, stream = &stream);

        let a = Array::from_int(5);
        let _a = repeat_all!(a, 3, dtype = f32);

        let a = Array::from_int(5);
        let _a = repeat_all!(a, 3, dtype = f32, stream = &stream);
    }

    #[test]
    fn test_tri() {
        let stream = StreamOrDevice::default();

        // We are just testing that the macros compile
        let _a = tri!(3);
        let _a = tri!(3, stream = &stream);
        let _a = tri!(3, dtype = f32);
        let _a = tri!(3, 4);
        let _a = tri!(3, dtype = f32, stream = &stream);
        let _a = tri!(3, 4, stream = &stream);
        let _a = tri!(3, 4, dtype = f32);
        let _a = tri!(3, 4, 1);
        let _a = tri!(3, 4, dtype = f32, stream = &stream);
        let _a = tri!(3, 4, 1, stream = &stream);
        let _a = tri!(3, 4, 1, dtype = f32);
        let _a = tri!(3, 4, 1, dtype = f32, stream = &stream);
    }
}
