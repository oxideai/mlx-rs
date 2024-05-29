/// See [`crate::ops::conv_general`] for details.
#[macro_export]
macro_rules! conv_general {
    ($a:expr, $weight:expr) => {
        $crate::ops::conv_general(
            $a.as_ref(),
            $weight.as_ref(),
            None,
            None,
            None,
            None,
            None,
            None,
        )
    };
    ($a:expr, $weight:expr, $strides:expr) => {
        $crate::ops::conv_general(
            $a.as_ref(),
            $weight.as_ref(),
            $strides,
            None,
            None,
            None,
            None,
            None,
        )
    };
    ($a:expr, $weight:expr, $strides:expr, $padding:expr) => {
        $crate::ops::conv_general(
            $a.as_ref(),
            $weight.as_ref(),
            $strides,
            $padding,
            None,
            None,
            None,
            None,
        )
    };
    ($a:expr, $weight:expr, $strides:expr, $padding:expr, $kernel_dilation:expr) => {
        $crate::ops::conv_general(
            $a.as_ref(),
            $weight.as_ref(),
            $strides,
            $padding,
            $kernel_dilation,
            None,
            None,
            None,
        )
    };
    ($a:expr, $weight:expr, $strides:expr, $padding:expr, $kernel_dilation:expr, $input_dilation:expr) => {
        $crate::ops::conv_general(
            $a.as_ref(),
            $weight.as_ref(),
            $strides,
            $padding,
            $kernel_dilation,
            $input_dilation,
            None,
            None,
        )
    };
    ($a:expr, $weight:expr, $strides:expr, $padding:expr, $kernel_dilation:expr, $input_dilation:expr, $groups:expr) => {
        $crate::ops::conv_general(
            $a.as_ref(),
            $weight.as_ref(),
            $strides,
            $padding,
            $kernel_dilation,
            $input_dilation,
            $groups,
            None,
        )
    };
    ($a:expr, $weight:expr, $strides:expr, $padding:expr, $kernel_dilation:expr, $input_dilation:expr, $groups:expr, $flip:expr) => {
        $crate::ops::conv_general(
            $a.as_ref(),
            $weight.as_ref(),
            $strides,
            $padding,
            $kernel_dilation,
            $input_dilation,
            $groups,
            $flip,
        )
    };
    ($a:expr, $weight:expr, $strides:expr, $padding:expr, $kernel_dilation:expr, $input_dilation:expr, $groups:expr, $flip:expr, stream=$stream:expr) => {
        $crate::ops::conv_general_device(
            $a.as_ref(),
            $weight.as_ref(),
            $strides,
            $padding,
            $kernel_dilation,
            $input_dilation,
            $groups,
            $flip,
            $stream,
        )
    };
}

/// See [`crate::ops::conv1d`] for details.
#[macro_export]
macro_rules! conv1d {
    ($a:expr, $weight:expr) => {
        $crate::ops::conv1d($a.as_ref(), $weight.as_ref(), None, None, None, None)
    };
    ($a:expr, $weight:expr, $stride:expr) => {
        $crate::ops::conv1d($a.as_ref(), $weight.as_ref(), $stride, None, None, None)
    };
    ($a:expr, $weight:expr, $stride:expr, $padding:expr) => {
        $crate::ops::conv1d($a.as_ref(), $weight.as_ref(), $stride, $padding, None, None)
    };
    ($a:expr, $weight:expr, $stride:expr, $padding:expr, $dilation:expr) => {
        $crate::ops::conv1d(
            $a.as_ref(),
            $weight.as_ref(),
            $stride,
            $padding,
            $dilation,
            None,
        )
    };
    ($a:expr, $weight:expr, $stride:expr, $padding:expr, $dilation:expr, $groups:expr) => {
        $crate::ops::conv1d(
            $a.as_ref(),
            $weight.as_ref(),
            $stride,
            $padding,
            $dilation,
            $groups,
        )
    };
    ($a:expr, $weight:expr, $stride:expr, $padding:expr, $dilation:expr, $groups:expr, stream=$stream:expr) => {
        $crate::ops::conv1d_device(
            $a.as_ref(),
            $weight.as_ref(),
            $stride,
            $padding,
            $dilation,
            $groups,
            $stream,
        )
    };
}

/// See [`crate::ops::conv2d`] for details.
#[macro_export]
macro_rules! conv2d {
    ($a:expr, $weight:expr) => {
        $crate::ops::conv2d($a.as_ref(), $weight.as_ref(), None, None, None, None)
    };
    ($a:expr, $weight:expr, $stride:expr) => {
        $crate::ops::conv2d($a.as_ref(), $weight.as_ref(), $stride, None, None, None)
    };
    ($a:expr, $weight:expr, $stride:expr, $padding:expr) => {
        $crate::ops::conv2d($a.as_ref(), $weight.as_ref(), $stride, $padding, None, None)
    };
    ($a:expr, $weight:expr, $stride:expr, $padding:expr, $dilation:expr) => {
        $crate::ops::conv2d(
            $a.as_ref(),
            $weight.as_ref(),
            $stride,
            $padding,
            $dilation,
            None,
        )
    };
    ($a:expr, $weight:expr, $stride:expr, $padding:expr, $dilation:expr, $groups:expr) => {
        $crate::ops::conv2d(
            $a.as_ref(),
            $weight.as_ref(),
            $stride,
            $padding,
            $dilation,
            $groups,
        )
    };
    ($a:expr, $weight:expr, $stride:expr, $padding:expr, $dilation:expr, $groups:expr, stream=$stream:expr) => {
        $crate::ops::conv2d_device(
            $a.as_ref(),
            $weight.as_ref(),
            $stride,
            $padding,
            $dilation,
            $groups,
            $stream,
        )
    };
}

#[cfg(test)]
mod tests {
    use crate::{Array, StreamOrDevice};

    #[test]
    fn test_conv_general() {
        let stream = StreamOrDevice::default();
        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let input_array = Array::from_slice(&input_data, &[1, 5, 2]);

        let weight_data = [0.5, 0.0, -0.5, 1.0, 0.0, 1.5, 2.0, 0.0, -2.0, 1.5, 0.0, 1.0];
        let weight_array = Array::from_slice(&weight_data, &[2, 3, 2]);

        // We are just testing that the macro compiles
        let _ = conv_general!(input_array, weight_array);
        let _ = conv_general!(input_array, weight_array, &[1][..]);
        let _ = conv_general!(input_array, weight_array, &[1][..], &[0][..]);
        let _ = conv_general!(input_array, weight_array, &[1][..], &[0][..], &[1][..]);
        let _ = conv_general!(
            input_array,
            weight_array,
            &[1][..],
            &[0][..],
            &[1][..],
            &[1][..]
        );
        let _ = conv_general!(
            input_array,
            weight_array,
            &[1][..],
            &[0][..],
            &[1][..],
            &[1][..],
            1
        );
        let _ = conv_general!(
            input_array,
            weight_array,
            &[1][..],
            &[0][..],
            &[1][..],
            &[1][..],
            1,
            false
        );
        let _ = conv_general!(
            input_array,
            weight_array,
            &[1][..],
            &[0][..],
            &[1][..],
            &[1][..],
            1,
            false,
            stream = stream
        );
    }

    #[test]
    fn test_conv1d() {
        let stream = StreamOrDevice::default();
        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let input_array = Array::from_slice(&input_data, &[1, 5, 2]);

        let weight_data = [0.5, 0.0, -0.5, 1.0, 0.0, 1.5, 2.0, 0.0, -2.0, 1.5, 0.0, 1.0];
        let weight_array = Array::from_slice(&weight_data, &[2, 3, 2]);

        // We are just testing that the macro compiles
        let _ = conv1d!(input_array, weight_array);
        let _ = conv1d!(input_array, weight_array, 1);
        let _ = conv1d!(input_array, weight_array, 1, 0);
        let _ = conv1d!(input_array, weight_array, 1, 0, 1);
        let _ = conv1d!(input_array, weight_array, 1, 0, 1, 1);
        let _ = conv1d!(input_array, weight_array, 1, 0, 1, 1, stream = stream);
    }

    #[test]
    fn test_conv2d() {
        let stream = StreamOrDevice::default();
        let input_data = [1.0, 2.0, 3.0, 4.0];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        // Define a 2x2 kernel with one input channel and one output channel
        let weight_data = [1.0, 0.0, 0.0, 1.0];
        let weight_shape = [1, 2, 2, 1]; // [C_out, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        // We are just testing that the macro compiles
        let _ = conv2d!(input_array, weight_array);
        let _ = conv2d!(input_array, weight_array, (1, 1));
        let _ = conv2d!(input_array, weight_array, (1, 1), (0, 0));
        let _ = conv2d!(input_array, weight_array, (1, 1), (0, 0), (1, 1));
        let _ = conv2d!(input_array, weight_array, (1, 1), (0, 0), (1, 1), 1);
        let _ = conv2d!(
            input_array,
            weight_array,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            stream = stream
        );
    }
}
