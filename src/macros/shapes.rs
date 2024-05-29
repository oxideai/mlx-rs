/// See [`crate::ops::as_strided`] for details.
#[macro_export]
macro_rules! as_strided {
    ($a:expr) => {
        $crate::ops::as_strided($a.as_ref(), None, None, None)
    };
    ($a:expr, $shape:expr) => {
        $crate::ops::as_strided($a.as_ref(), $shape, None, None)
    };
    ($a:expr, $shape:expr, $strides:expr) => {
        $crate::ops::as_strided($a.as_ref(), $shape, $strides, None)
    };
    ($a:expr, $shape:expr, $strides:expr, $offset:expr) => {
        $crate::ops::as_strided($a.as_ref(), $shape, $strides, $offset)
    };
    ($a:expr, $shape:expr, $strides:expr, $offset:expr, stream=$stream:expr) => {
        $crate::ops::as_strided_device($a.as_ref(), $shape, $strides, $offset, $stream)
    };
}

/// See [`crate::ops::broadcast_to`] for details.
#[macro_export]
macro_rules! broadcast_to {
    ($a:expr, $shape:expr) => {
        $crate::ops::broadcast_to($a.as_ref(), $shape)
    };
    ($a:expr, $shape:expr, stream=$stream:expr) => {
        $crate::ops::broadcast_to_device($a.as_ref(), $shape, $stream)
    };
}

/// See [`crate::ops::concatenate`] for details.
#[macro_export]
macro_rules! concatenate {
    ($arrays:expr) => {
        $crate::ops::concatenate($arrays, None)
    };
    ($arrays:expr, $axis:expr) => {
        $crate::ops::concatenate($arrays, $axis)
    };
    ($arrays:expr, $axis:expr, stream=$stream:expr) => {
        $crate::ops::concatenate_device($arrays, $axis, $stream)
    };
}

/// See [`crate::ops::expand_dims`] for details.
#[macro_export]
macro_rules! expand_dims {
    ($a:expr, $axis:expr) => {
        $crate::ops::expand_dims($a.as_ref(), $axis)
    };
    ($a:expr, $axis:expr, stream=$stream:expr) => {
        $crate::ops::expand_dims_device($a.as_ref(), $axis, $stream)
    };
}

/// See [`crate::ops::flatten`] for details.
#[macro_export]
macro_rules! flatten {
    ($a:expr) => {
        $crate::ops::flatten($a.as_ref(), None, None)
    };
    ($a:expr, $start:expr) => {
        $crate::ops::flatten($a.as_ref(), $start, None)
    };
    ($a:expr, $start:expr, $end:expr) => {
        $crate::ops::flatten($a.as_ref(), $start, $end)
    };
    ($a:expr, $start:expr, $end:expr, stream=$stream:expr) => {
        $crate::ops::flatten_device($a.as_ref(), $start, $end, $stream)
    };
}

/// See [`crate::ops::reshape`] for details.
#[macro_export]
macro_rules! reshape {
    ($a:expr, $shape:expr) => {
        $crate::ops::reshape($a.as_ref(), $shape)
    };
    ($a:expr, $shape:expr, stream=$stream:expr) => {
        $crate::ops::reshape_device($a.as_ref(), $shape, $stream)
    };
}

/// See [`crate::ops::squeeze`] for details.
#[macro_export]
macro_rules! squeeze {
    ($a:expr) => {
        $crate::ops::squeeze($a.as_ref(), None)
    };
    ($a:expr, $axis:expr) => {
        $crate::ops::squeeze($a.as_ref(), $axis)
    };
    ($a:expr, $axis:expr, stream=$stream:expr) => {
        $crate::ops::squeeze_device($a.as_ref(), $axis, $stream)
    };
}

/// See [`crate::ops::at_least_1d`] for details.
#[macro_export]
macro_rules! at_least_1d {
    ($a:expr) => {
        $crate::ops::at_least_1d($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::ops::at_least_1d_device($a.as_ref(), $stream)
    };
}

/// See [`crate::ops::at_least_2d`] for details.
#[macro_export]
macro_rules! at_least_2d {
    ($a:expr) => {
        $crate::ops::at_least_2d($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::ops::at_least_2d_device($a.as_ref(), $stream)
    };
}

/// See [`crate::ops::at_least_3d`] for details.
#[macro_export]
macro_rules! at_least_3d {
    ($a:expr) => {
        $crate::ops::at_least_3d($a)
    };
    ($a:expr, stream=$stream:expr) => {
        $crate::ops::at_least_3d_device($a.as_ref(), $stream)
    };
}

/// See [`crate::ops::move_axis`] for details.
#[macro_export]
macro_rules! move_axis {
    ($a:expr, $src:expr, $dst:expr) => {
        $crate::ops::move_axis($a.as_ref(), $src, $dst)
    };
    ($a:expr, $src:expr, $dst:expr, stream=$stream:expr) => {
        $crate::ops::move_axis_device($a.as_ref(), $src, $dst, $stream)
    };
}

/// See [`crate::ops::split`] for details.
#[macro_export]
macro_rules! split {
    ($a:expr, $indices:expr) => {
        $crate::ops::split($a.as_ref(), $indices, None)
    };
    ($a:expr, $indices:expr, $axis:expr) => {
        $crate::ops::split($a.as_ref(), $indices, $axis)
    };
    ($a:expr, $indices:expr, $axis:expr, stream=$stream:expr) => {
        $crate::ops::split_device($a.as_ref(), $indices, $axis, $stream)
    };
}

/// See [`crate::ops::split_equal`] for details.
#[macro_export]
macro_rules! split_equal {
    ($a:expr, $num_splits:expr) => {
        $crate::ops::split_equal($a.as_ref(), $num_splits, None)
    };
    ($a:expr, $num_splits:expr, $axis:expr) => {
        $crate::ops::split_equal($a.as_ref(), $num_splits, $axis)
    };
    ($a:expr, $num_splits:expr, $axis:expr, stream=$stream:expr) => {
        $crate::ops::split_equal_device($a.as_ref(), $num_splits, $axis, $stream)
    };
}

/// See [`crate::ops::pad`] for details.
#[macro_export]
macro_rules! pad {
    ($a:expr, $width:expr) => {
        $crate::ops::pad($a.as_ref(), $width, None)
    };
    ($a:expr, $width:expr, $value:expr) => {
        $crate::ops::pad($a.as_ref(), $width, $value)
    };
    ($a:expr, $width:expr, $value:expr, stream=$stream:expr) => {
        $crate::ops::pad_device($a.as_ref(), $width, $value, $stream)
    };
}

/// See [`crate::ops::stack`] for details.
#[macro_export]
macro_rules! stack {
    ($arrays:expr, $axis:expr) => {
        $crate::ops::stack($arrays, $axis)
    };
    ($arrays:expr, $axis:expr, stream=$stream:expr) => {
        $crate::ops::stack_device($arrays, $axis, $stream)
    };
}

/// See [`crate::ops::stack_all`] for details.
#[macro_export]
macro_rules! stack_all {
    ($arrays:expr) => {
        $crate::ops::stack_all($arrays)
    };
    ($arrays:expr, stream=$stream:expr) => {
        $crate::ops::stack_all_device($arrays, $stream)
    };
}

/// See [`crate::ops::swap_axes`] for details.
#[macro_export]
macro_rules! swap_axes {
    ($a:expr, $axis1:expr, $axis2:expr) => {
        $crate::ops::swap_axes($a.as_ref(), $axis1, $axis2)
    };
    ($a:expr, $axis1:expr, $axis2:expr, stream=$stream:expr) => {
        $crate::ops::swap_axes_device($a.as_ref(), $axis1, $axis2, $stream)
    };
}

/// See [`crate::ops::tile`] for details.
#[macro_export]
macro_rules! tile {
    ($a:expr, $reps:expr) => {
        $crate::ops::tile($a.as_ref(), $reps)
    };
    ($a:expr, $reps:expr, stream=$stream:expr) => {
        $crate::ops::tile_device($a.as_ref(), $reps, $stream)
    };
}

/// See [`crate::ops::transpose`] for details.
#[macro_export]
macro_rules! transpose {
    ($a:expr) => {
        $crate::ops::transpose($a.as_ref(), None)
    };
    ($a:expr, $axes:expr) => {
        $crate::ops::transpose($a.as_ref(), $axes)
    };
    ($a:expr, $axes:expr, stream=$stream:expr) => {
        $crate::ops::transpose_device($a.as_ref(), $axes, $stream)
    };
}

#[cfg(test)]
mod tests {
    use crate::{Array, StreamOrDevice};

    #[test]
    fn test_as_strided() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[10]);

        // We are just testing that the macro compiles
        let _y = as_strided!(&x);
        let _y = as_strided!(&x, &[3, 3][..]);
        let _y = as_strided!(&x, &[3, 3][..], &[1, 1][..]);
        let _y = as_strided!(&x, &[3, 3][..], &[1, 1][..], 0);
        let _y = as_strided!(&x, &[3, 3][..], &[1, 1][..], 0, stream=stream);
    }

    #[test]
    fn test_broadcast_to() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..9, &[9]);

        // We are just testing that the macro compiles
        let _y = broadcast_to!(&x, &[9][..]);
        let _y = broadcast_to!(&x, &[9][..], stream=stream);
    }

    #[test]
    fn test_concatenate() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[10]);

        // We are just testing that the macro compiles
        let _y = concatenate!(&[&x, &x]);
        let _y = concatenate!(&[&x, &x], 0);
        let _y = concatenate!(&[&x, &x], 0, stream=stream);
    }

    #[test]
    fn test_expand_dims() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[10]);

        // We are just testing that the macro compiles
        let _y = expand_dims!(&x, &[0]);
        let _y = expand_dims!(&x, &[0], stream=stream);
    }

    #[test]
    fn test_flatten() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[10]);

        // We are just testing that the macro compiles
        let _y = flatten!(&x);
        let _y = flatten!(&x, 0);
        let _y = flatten!(&x, 0, 1);
        let _y = flatten!(&x, 0, 1, stream=stream);
    }

    #[test]
    fn test_reshape() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..9, &[9]);

        // We are just testing that the macro compiles
        let _y = reshape!(&x, &[3, 3]);
        let _y = reshape!(&x, &[3, 3], stream=stream);
    }

    #[test]
    fn test_squeeze() {
        let stream = StreamOrDevice::default();
        let x = Array::zeros::<i32>(&[1, 2, 1, 3]);

        // We are just testing that the macro compiles
        let _y = squeeze!(&x);
        let _y = squeeze!(&x, &[0, 2][..]);
        let _y = squeeze!(&x, &[0, 2][..], stream=stream);
    }

    #[test]
    fn test_at_least_1d() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[10]);

        // We are just testing that the macro compiles
        let _y = at_least_1d!(&x);
        let _y = at_least_1d!(&x, stream=stream);
    }

    #[test]
    fn test_at_least_2d() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[10]);

        // We are just testing that the macro compiles
        let _y = at_least_2d!(&x);
        let _y = at_least_2d!(&x, stream=stream);
    }

    #[test]
    fn test_at_least_3d() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[10]);

        // We are just testing that the macro compiles
        let _y = at_least_3d!(&x);
        let _y = at_least_3d!(&x, stream=stream);
    }

    #[test]
    fn test_move_axis() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[2, 5]);

        // We are just testing that the macro compiles
        let _y = move_axis!(&x, 0, 1);
        let _y = move_axis!(&x, 0, 1, stream=stream);
    }

    #[test]
    fn test_split() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[10]);

        // We are just testing that the macro compiles
        let _y = split!(&x, &[3]);
        let _y = split!(&x, &[3], 0);
        let _y = split!(&x, &[3], 0, stream=stream);
    }

    #[test]
    fn test_split_equal() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..9, &[9]);

        // We are just testing that the macro compiles
        let _y = split_equal!(&x, 3);
        let _y = split_equal!(&x, 3, 0);
        let _y = split_equal!(&x, 3, 0, stream=stream);
    }

    #[test]
    fn test_pad() {
        let stream = StreamOrDevice::default();
        let x = Array::zeros::<f32>(&[1, 2, 3]);

        // We are just testing that the macro compiles
        pad!(&x, 1);
        pad!(&x, (0, 1), Array::from_int(1));
        pad!(&x, (0, 1), Array::from_int(1), stream=stream);
    }

    #[test]
    fn test_stack() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[10]);

        // We are just testing that the macro compiles
        stack!(&[&x, &x], 0);
        stack!(&[&x, &x], 0, stream=stream);
    }

    #[test]
    fn test_stack_all() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[10]);

        // We are just testing that the macro compiles
        stack_all!(&[&x, &x]);
        stack_all!(&[&x, &x], stream=stream);
    }

    #[test]
    fn test_swap_axes() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[2, 5]);

        // We are just testing that the macro compiles
        swap_axes!(&x, 0, 1);
        swap_axes!(&x, 0, 1, stream=stream);
    }

    #[test]
    fn test_tile() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[2, 5]);

        // We are just testing that the macro compiles
        tile!(&x, &[2, 3]);
        tile!(&x, &[2, 3], stream=stream);
    }

    #[test]
    fn test_transpose() {
        let stream = StreamOrDevice::default();
        let x = Array::from_iter(0..10, &[2, 5]);

        // We are just testing that the macro compiles
        transpose!(&x);
        transpose!(&x, &[1, 0][..]);
        transpose!(&x, &[1, 0][..], stream=stream);
    }
}
