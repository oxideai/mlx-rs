use crate::error::OperationError;
use crate::{Array, StreamOrDevice};
use mlx_macros::default_device;

#[inline]
#[allow(clippy::too_many_arguments)]
fn conv_general_device_inner(
    array: &Array,
    weight: &Array,
    strides: &[i32],
    padding_lo: &[i32],
    padding_hi: &[i32],
    kernel_dilation: &[i32],
    input_dilation: &[i32],
    groups: i32,
    flip: bool,
    stream: StreamOrDevice,
) -> Array {
    unsafe {
        Array::from_ptr(mlx_sys::mlx_conv_general(
            array.as_ptr(),
            weight.as_ptr(),
            strides.as_ptr(),
            strides.len(),
            padding_lo.as_ptr(),
            padding_lo.len(),
            padding_hi.as_ptr(),
            padding_hi.len(),
            kernel_dilation.as_ptr(),
            kernel_dilation.len(),
            input_dilation.as_ptr(),
            input_dilation.len(),
            groups,
            flip,
            stream.as_ptr(),
        ))
    }
}

/// General convolution over an input with several channels.
///
/// - Only 1d and 2d convolutions are supported at the moment
/// - the default `groups: 1` is currently supported
///
/// # Params
/// - array: Input array of shape `&[N, ..., C_in]`
/// - weight: Weight array of shape `&[C_out, ..., C_in]`
/// - strides: The kernel strides. All dimensions get the same stride if only one number is specified.
/// - padding: The input padding. All dimensions get the same padding if only one number is specified.
/// - kernel_dilation: The kernel dilation. All dimensions get the same dilation if only one number is specified.
/// - input_dilation: The input dilation. All dimensions get the same dilation if only one number is specified.
/// - groups: Input feature groups
/// - flip: Flip the order in which the spatial dimensions of the weights are processed.
///   Performs the cross-correlation operator when `flip` is `false` and the convolution
///   operator otherwise.
/// - stream: stream or device to evaluate on
#[default_device]
#[allow(clippy::too_many_arguments)]
pub fn conv_general_device<'a>(
    array: &Array,
    weight: &Array,
    strides: impl Into<Option<&'a [i32]>>,
    padding: impl Into<Option<&'a [i32]>>,
    kernel_dilation: impl Into<Option<&'a [i32]>>,
    input_dilation: impl Into<Option<&'a [i32]>>,
    groups: impl Into<Option<i32>>,
    flip: impl Into<Option<bool>>,
    stream: StreamOrDevice,
) -> Array {
    try_conv_general_device(
        array,
        weight,
        strides,
        padding,
        kernel_dilation,
        input_dilation,
        groups,
        flip,
        stream,
    )
    .unwrap()
}

/// General convolution over an input with several channels without validating the inputs.
///
/// - Only 1d and 2d convolutions are supported at the moment
/// - the default `groups: 1` is currently supported
///
/// # Params
/// - array: Input array of shape `&[N, ..., C_in]`
/// - weight: Weight array of shape `&[C_out, ..., C_in]`
/// - strides: The kernel strides. All dimensions get the same stride if only one number is specified.
/// - padding: The input padding. All dimensions get the same padding if only one number is specified.
/// - kernel_dilation: The kernel dilation. All dimensions get the same dilation if only one number is specified.
/// - input_dilation: The input dilation. All dimensions get the same dilation if only one number is specified.
/// - groups: Input feature groups
/// - flip: Flip the order in which the spatial dimensions of the weights are processed.
///   Performs the cross-correlation operator when `flip` is `false` and the convolution
///   operator otherwise.
/// - stream: stream or device to evaluate on
///
/// # Safety
///
/// This function is unsafe because it does not validate:
/// - if the groups are not equal to 1
/// - if the input and weight arrays do not have the same number of input channels
/// - if the input and weight arrays do not have the same number of spatial dimensions
/// - if arrays are not floating point arrays
/// - if arrays are not 1d or 2d arrays
#[default_device]
#[allow(clippy::too_many_arguments)]
pub unsafe fn conv_general_device_unchecked<'a>(
    array: &Array,
    weight: &Array,
    strides: impl Into<Option<&'a [i32]>>,
    padding: impl Into<Option<&'a [i32]>>,
    kernel_dilation: impl Into<Option<&'a [i32]>>,
    input_dilation: impl Into<Option<&'a [i32]>>,
    groups: impl Into<Option<i32>>,
    flip: impl Into<Option<bool>>,
    stream: StreamOrDevice,
) -> Array {
    let strides = strides.into().unwrap_or(&[1]);
    let padding = padding.into().unwrap_or(&[0]);
    let kernel_dilation = kernel_dilation.into().unwrap_or(&[1]);
    let input_dilation = input_dilation.into().unwrap_or(&[1]);
    let groups = groups.into().unwrap_or(1);
    let flip = flip.into().unwrap_or(false);

    conv_general_device_inner(
        array,
        weight,
        strides,
        padding,
        padding,
        kernel_dilation,
        input_dilation,
        groups,
        flip,
        stream,
    )
}

/// General convolution over an input with several channels returning an error if the inputs are invalid.
///
/// - Only 1d and 2d convolutions are supported at the moment
/// - the default `groups: 1` is currently supported
///
/// # Params
/// - array: Input array of shape `&[N, ..., C_in]`
/// - weight: Weight array of shape `&[C_out, ..., C_in]`
/// - strides: The kernel strides. All dimensions get the same stride if only one number is specified.
/// - padding: The input padding. All dimensions get the same padding if only one number is specified.
/// - kernel_dilation: The kernel dilation. All dimensions get the same dilation if only one number is specified.
/// - input_dilation: The input dilation. All dimensions get the same dilation if only one number is specified.
/// - groups: Input feature groups
/// - flip: Flip the order in which the spatial dimensions of the weights are processed.
///   Performs the cross-correlation operator when `flip` is `false` and the convolution
///   operator otherwise.
/// - stream: stream or device to evaluate on
#[default_device]
#[allow(clippy::too_many_arguments)]
pub fn try_conv_general_device<'a>(
    array: &Array,
    weight: &Array,
    strides: impl Into<Option<&'a [i32]>>,
    padding: impl Into<Option<&'a [i32]>>,
    kernel_dilation: impl Into<Option<&'a [i32]>>,
    input_dilation: impl Into<Option<&'a [i32]>>,
    groups: impl Into<Option<i32>>,
    flip: impl Into<Option<bool>>,
    stream: StreamOrDevice,
) -> Result<Array, OperationError> {
    let strides = strides.into().unwrap_or(&[1]);
    let padding = padding.into().unwrap_or(&[0]);
    let kernel_dilation = kernel_dilation.into().unwrap_or(&[1]);
    let input_dilation = input_dilation.into().unwrap_or(&[1]);
    let groups = groups.into().unwrap_or(1);
    let flip = flip.into().unwrap_or(false);

    if groups != 1 {
        return Err(OperationError::WrongInput(
            "groups != 1 are not supported".to_string(),
        ));
    }

    let spatial_dims = array.ndim().checked_sub(2);
    if spatial_dims.is_none() {
        return Err(OperationError::WrongInput(
            "Only 1d and 2d arrays are supported.".to_string(),
        ));
    }

    let spatial_dims = spatial_dims.unwrap();

    if !(1..=2).contains(&spatial_dims) {
        return Err(OperationError::WrongInput(
            "Only 1d and 2d arrays are supported.".to_string(),
        ));
    }

    if !array.dtype().is_float() {
        return Err(OperationError::WrongInput(
            "Only floating point arrays are supported".to_string(),
        ));
    }

    if array.ndim() != spatial_dims + 2 {
        return Err(OperationError::WrongInput(
            format!("Invalid input array with {} dimensions for {}d convolution. Expected array with {} dimensions", array.ndim(), spatial_dims, spatial_dims + 2),
        ));
    }

    if weight.ndim() != spatial_dims + 2 {
        return Err(OperationError::WrongInput(
            format!("Invalid weight array with {} dimensions for {}d convolution. Expected array with {} dimensions", weight.ndim(), spatial_dims, spatial_dims + 2),
        ));
    }

    if array.shape().get(spatial_dims + 1) != weight.shape().get(spatial_dims + 1) {
        return Err(OperationError::WrongDimensions(
            "Input and weight arrays must have the same number of input channels".to_string(),
        ));
    }

    Ok(conv_general_device_inner(
        array,
        weight,
        strides,
        padding,
        padding,
        kernel_dilation,
        input_dilation,
        groups,
        flip,
        stream,
    ))
}

/// 1D convolution over an input with several channels.
///
/// > Only the default `groups=1` is currently supported.
///
/// # Params
/// - array: input array of shape `&[N, H, C_in]`
/// - weight: weight array of shape `&[C_out, H, C_in]`
/// - stride: kernel stride
/// - padding: input padding
/// - dilation: kernel dilation
/// - groups: input feature groups
/// - stream: stream or device to evaluate on
#[default_device]
pub fn conv1d_device(
    array: &Array,
    weight: &Array,
    stride: impl Into<Option<i32>>,
    padding: impl Into<Option<i32>>,
    dilation: impl Into<Option<i32>>,
    groups: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    try_conv1d_device(array, weight, stride, padding, dilation, groups, stream).unwrap()
}

/// 1D convolution over an input with several channels returning an error if the inputs are invalid.
///
/// > Only the default `groups=1` is currently supported.
///
/// # Params
/// - array: input array of shape `&[N, H, C_in]`
/// - weight: weight array of shape `&[C_out, H, C_in]`
/// - stride: kernel stride
/// - padding: input padding
/// - dilation: kernel dilation
/// - groups: input feature groups
/// - stream: stream or device to evaluate on
#[default_device]
pub fn try_conv1d_device(
    array: &Array,
    weight: &Array,
    stride: impl Into<Option<i32>>,
    padding: impl Into<Option<i32>>,
    dilation: impl Into<Option<i32>>,
    groups: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Result<Array, OperationError> {
    let stride = stride.into().unwrap_or(1);
    let padding = padding.into().unwrap_or(0);
    let dilation = dilation.into().unwrap_or(1);
    let groups = groups.into().unwrap_or(1);

    try_conv_general_device(
        array,
        weight,
        &[stride][..],
        &[padding][..],
        &[dilation][..],
        None,
        groups,
        false,
        stream,
    )
}

/// 2D convolution over an input with several channels.
///
/// > Only the default `groups=1` is currently supported.
///
/// # Params
/// - array: input array of shape `[N, H, W, C_in]`
/// - weight: weight array of shape `[C_out, H, W, C_in]`
/// - stride: kernel stride
/// - padding: input padding
/// - dilation: kernel dilation
/// - groups: input feature groups
/// - stream: stream or device to evaluate on
#[default_device]
pub fn conv2d_device(
    array: &Array,
    weight: &Array,
    stride: impl Into<Option<(i32, i32)>>,
    padding: impl Into<Option<(i32, i32)>>,
    dilation: impl Into<Option<(i32, i32)>>,
    groups: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    try_conv2d_device(array, weight, stride, padding, dilation, groups, stream).unwrap()
}

/// 2D convolution over an input with several channels returning an error if the inputs are invalid.
///
/// > Only the default `groups=1` is currently supported.
///
/// # Params
/// - array: input array of shape `[N, H, W, C_in]`
/// - weight: weight array of shape `[C_out, H, W, C_in]`
/// - stride: kernel stride
/// - padding: input padding
/// - dilation: kernel dilation
/// - groups: input feature groups
/// - stream: stream or device to evaluate on
#[default_device]
pub fn try_conv2d_device(
    array: &Array,
    weight: &Array,
    stride: impl Into<Option<(i32, i32)>>,
    padding: impl Into<Option<(i32, i32)>>,
    dilation: impl Into<Option<(i32, i32)>>,
    groups: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Result<Array, OperationError> {
    let stride = stride.into().unwrap_or((1, 1));
    let padding = padding.into().unwrap_or((0, 0));
    let dilation = dilation.into().unwrap_or((1, 1));

    try_conv_general_device(
        array,
        weight,
        &[stride.0, stride.1][..],
        &[padding.0, padding.1][..],
        &[dilation.0, dilation.1][..],
        None,
        groups,
        false,
        stream,
    )
}

// TODO: Implement convolve once we have `reshape` and `slice`

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_conv1d_complex_device() {
        // Define a 1D input with two channels
        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let input_array = Array::from_slice(&input_data, &[1, 5, 2]);

        // Define a 1D kernel with two input channels and two output channels
        let weight_data = [0.5, 0.0, -0.5, 1.0, 0.0, 1.5, 2.0, 0.0, -2.0, 1.5, 0.0, 1.0];
        let weight_array = Array::from_slice(&weight_data, &[2, 3, 2]);

        let mut result = conv1d(
            &input_array,
            &weight_array,
            Some(1), // stride
            Some(0), // padding
            Some(1), // dilation
            Some(1), // groups
        );

        result.eval();
        let expected_output = [12.0, 8.0, 17.0, 13.0, 22.0, 18.0];
        assert_eq!(result.shape(), &[1, 3, 2]);
        assert_eq!(result.as_slice::<f32>(), &expected_output);
    }

    #[test]
    fn test_conv2d() {
        // Define a 2x2 input with one channel (grayscale image or similar)
        let input_data = [1.0, 2.0, 3.0, 4.0];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        // Define a 2x2 kernel with one input channel and one output channel
        let weight_data = [1.0, 0.0, 0.0, 1.0];
        let weight_shape = [1, 2, 2, 1]; // [C_out, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        // Perform the convolution with no padding and stride of 1
        let mut result = conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(1),      // groups
        );

        result.eval();
        // Expected result is the convolution of a 2x2 filter over a 2x2 input with valid padding, resulting in a single output value
        let expected_output = 1.0 * 1.0 + 2.0 * 0.0 + 3.0 * 0.0 + 4.0 * 1.0; // = 1*1 + 4*1 = 5
        assert_eq!(result.as_slice::<f32>(), &[expected_output]);
    }

    #[test]
    fn test_conv_wrong_dimensions() {
        let input_data = [1.0, 2.0, 3.0, 4.0];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        let weight_data = [1.0, 0.0, 0.0, 1.0];
        let weight_shape = [1, 2, 2]; // [C_out, H_k, W_k]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        let result = try_conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(1),      // groups
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_conv_invalid_group_size() {
        let input_data = [1.0, 2.0, 3.0, 4.0];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        let weight_data = [1.0, 0.0, 0.0, 1.0];
        let weight_shape = [1, 2, 2, 1]; // [C_out, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        let result = try_conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(2),      // groups
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_conv_non_float() {
        let input_data = [1, 2, 3, 4];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        let weight_data = [1, 0, 0, 1];
        let weight_shape = [1, 2, 2, 1]; // [C_out, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        let result = try_conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(1),      // groups
        );

        assert!(result.is_err());
    }
}
