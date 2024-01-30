use cxx::UniquePtr;

use crate::{array::ffi::array, Optional};

pub type OptionalArray = Optional<UniquePtr<array>>;

unsafe impl cxx::ExternType for OptionalArray {
    type Id = cxx::type_id!("mlx_cxx::OptionalArray");

    type Kind = cxx::kind::Opaque;
}

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx-cxx/ops.hpp");

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        #[namespace = "mlx_cxx"]
        type StreamOrDevice = crate::StreamOrDevice;

        #[namespace = "mlx::core"]
        type Dtype = crate::dtype::ffi::Dtype;

        #[namespace = "mlx::core"]
        type float16_t = crate::types::float16::float16_t;

        #[namespace = "mlx::core"]
        type bfloat16_t = crate::types::bfloat16::bfloat16_t;

        #[namespace = "mlx::core"]
        type complex64_t = crate::types::complex64::complex64_t;

        #[namespace = "mlx_cxx"]
        type OptionalArray = crate::ops::OptionalArray;

        // A 1D std::unique_ptr<mlx::core::array> of numbers starting at `start` (optional),
        // stopping at stop, stepping by `step` (optional).

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_f64"]
        fn arange(start: f64, stop: f64, step: f64, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_start_stop_dtype"]
        fn arange(
            start: f64,
            stop: f64,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_start_stop_f64"]
        fn arange(start: f64, stop: f64, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_stop_dtype"]
        fn arange(stop: f64, dtype: Dtype, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_stop_f64"]
        fn arange(stop: f64, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_i32"]
        fn arange(start: i32, stop: i32, step: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_start_stop_i32"]
        fn arange(start: i32, stop: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_stop_i32"]
        fn arange(stop: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        // A 1D std::unique_ptr<mlx::core::array> of `num` evenly spaced numbers in the range
        // `[start, stop]`
        #[namespace = "mlx_cxx"]
        fn linspace(
            start: f64,
            stop: f64,
            num: i32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        // Convert an array to the given data type.
        #[namespace = "mlx_cxx"]
        fn astype(a: &array, dtype: Dtype, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        // Create a view of an array with the given shape and strides.
        #[namespace = "mlx_cxx"]
        fn as_strided(
            a: &array,
            shape: UniquePtr<CxxVector<i32>>,
            strides: UniquePtr<CxxVector<usize>>,
            offset: usize,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn copy(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "full_vals_dtype"]
        fn full(
            shape: &CxxVector<i32>,
            vals: &array,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "full_vals"]
        fn full(
            shape: &CxxVector<i32>,
            vals: &array,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_bool_val_dtype(
            shape: &CxxVector<i32>,
            val: bool,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_uint8_val_dtype(
            shape: &CxxVector<i32>,
            val: u8,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_uint16_val_dtype(
            shape: &CxxVector<i32>,
            val: u16,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_uint32_val_dtype(
            shape: &CxxVector<i32>,
            val: u32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_uint64_val_dtype(
            shape: &CxxVector<i32>,
            val: u64,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_int8_val_dtype(
            shape: &CxxVector<i32>,
            val: i8,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_int16_val_dtype(
            shape: &CxxVector<i32>,
            val: i16,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_int32_val_dtype(
            shape: &CxxVector<i32>,
            val: i32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_int64_val_dtype(
            shape: &CxxVector<i32>,
            val: i64,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_float16_val_dtype(
            shape: &CxxVector<i32>,
            val: float16_t,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_bfloat16_val_dtype(
            shape: &CxxVector<i32>,
            val: bfloat16_t,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_float32_val_dtype(
            shape: &CxxVector<i32>,
            val: f32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_complex64_val_dtype(
            shape: &CxxVector<i32>,
            val: complex64_t,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_bool_val(
            shape: &CxxVector<i32>,
            val: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_uint8_val(
            shape: &CxxVector<i32>,
            val: u8,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_uint16_val(
            shape: &CxxVector<i32>,
            val: u16,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_uint32_val(
            shape: &CxxVector<i32>,
            val: u32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_uint64_val(
            shape: &CxxVector<i32>,
            val: u64,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_int8_val(
            shape: &CxxVector<i32>,
            val: i8,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_int16_val(
            shape: &CxxVector<i32>,
            val: i16,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_int32_val(
            shape: &CxxVector<i32>,
            val: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_int64_val(
            shape: &CxxVector<i32>,
            val: i64,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_float16_val(
            shape: &CxxVector<i32>,
            val: float16_t,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_bfloat16_val(
            shape: &CxxVector<i32>,
            val: bfloat16_t,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_float32_val(
            shape: &CxxVector<i32>,
            val: f32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn full_complex64_val(
            shape: &CxxVector<i32>,
            val: complex64_t,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "zeros_dtype"]
        fn zeros(shape: &CxxVector<i32>, dtype: Dtype, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn zeros(shape: &CxxVector<i32>, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn zeros_like(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ones_dtype"]
        fn ones(shape: &CxxVector<i32>, dtype: Dtype, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn ones(shape: &CxxVector<i32>, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn ones_like(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "eye_n_m_k_dtype"]
        fn eye(n: i32, m: i32, k: i32, dtype: Dtype, s: StreamOrDevice)
            -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "eye_n_dtype"]
        fn eye(n: i32, dtype: Dtype, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "eye_n_m"]
        fn eye(n: i32, m: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "eye_n_m_k"]
        fn eye(n: i32, m: i32, k: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "eye_n"]
        fn eye(n: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "identity_dtype"]
        fn identity(n: i32, dtype: Dtype, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn identity(n: i32, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "tri_n_m_k_dtype"]
        fn tri(n: i32, m: i32, k: i32, dtype: Dtype, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "tri_n_dtype"]
        fn tri(n: i32, dtype: Dtype, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn tril(x: UniquePtr<array>, k: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn triu(x: UniquePtr<array>, k: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn reshape(
            a: &array,
            shape: UniquePtr<CxxVector<i32>>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "flatten_start_axis_end_axis"]
        fn flatten(
            a: &array,
            start_axis: i32,
            end_axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn flatten(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "squeeze_axes"]
        fn squeeze(a: &array, axes: &CxxVector<i32>, s: StreamOrDevice)
            -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "squeeze_axis"]
        fn squeeze(a: &array, axis: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn squeeze(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "expand_dims_at_axes"]
        fn expand_dims(
            a: &array,
            axes: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "expand_dims_at_axis"]
        fn expand_dims(a: &array, axis: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "slice_start_stop_strides"]
        fn slice(
            a: &array,
            start: UniquePtr<CxxVector<i32>>,
            stop: UniquePtr<CxxVector<i32>>,
            strides: UniquePtr<CxxVector<i32>>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn slice(
            a: &array,
            start: &CxxVector<i32>,
            stop: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "split_n_at_axis"]
        fn split(
            a: &array,
            num_splits: i32,
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<CxxVector<array>>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "split_n"]
        fn split(
            a: &array,
            num_splits: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<CxxVector<array>>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "split_at_indices_along_axis"]
        fn split(
            a: &array,
            indices: &CxxVector<i32>,
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<CxxVector<array>>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "split_at_indices"]
        fn split(
            a: &array,
            indices: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<CxxVector<array>>>;

        #[namespace = "mlx_cxx"]
        fn clip(
            a: &array,
            a_min: &OptionalArray,
            a_max: &OptionalArray,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "concatenate_along_axis"]
        fn concatenate(
            arrays: &[UniquePtr<array>],
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn concatenate(arrays: &[UniquePtr<array>], s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "stack_along_axis"]
        fn stack(
            arrays: &[UniquePtr<array>],
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn stack(arrays: &[UniquePtr<array>], s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "repeat_along_axis"]
        fn repeat(
            arr: &array,
            repeats: i32,
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn repeat(arr: &array, repeats: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn tile(
            arr: &array,
            reps: UniquePtr<CxxVector<i32>>,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "transpose_axes"]
        fn transpose(
            a: &array,
            axes: UniquePtr<CxxVector<i32>>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn swapaxes(
            a: &array,
            axis1: i32,
            axis2: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn moveaxis(
            a: &array,
            source: i32,
            destination: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "pad_axes"]
        fn pad(
            a: &array,
            axes: &CxxVector<i32>,
            low_pad_size: &CxxVector<i32>,
            high_pad_size: &CxxVector<i32>,
            pad_value: &array,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "pad_unique_widths_for_each_axis"]
        fn pad(
            a: &array,
            pad_width: &[[i32; 2]],
            pad_value: &array,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "pad_same_widths_for_each_axis"]
        fn pad(
            a: &array,
            pad_width: &[i32; 2],
            pad_value: &array,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn pad(
            a: &array,
            pad_width: i32,
            pad_value: &array,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn transpose(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn broadcast_to(
            a: &array,
            shape: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn broadcast_arrays(
            inputs: &[UniquePtr<array>],
            s: StreamOrDevice,
        ) -> Result<UniquePtr<CxxVector<array>>>;

        #[namespace = "mlx_cxx"]
        fn equal(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn not_equal(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn greater(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn greater_equal(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn less(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn less_equal(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        /// True if two arrays have the same shape and elements.
        #[namespace = "mlx_cxx"]
        #[rust_name = "array_equal_equal_nan"]
        fn array_equal(
            a: &array,
            b: &array,
            equal_nan: bool,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_equal(a: &array, b: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn isnan(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn isinf(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn isposinf(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn isneginf(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        /// Renamed to `where_condition` because `where` is a reserved keyword in Rust.
        #[namespace = "mlx_cxx"]
        #[cxx_name = "where"]
        fn where_condition(
            condition: &array,
            x: &array,
            y: &array,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        /// True if all elements in the std::unique_ptr<mlx::core::array> are true (or non-zero).
        #[namespace = "mlx_cxx"]
        #[rust_name = "all_keepdims"]
        fn all(a: &array, keepdims: bool, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        /// True if all elements in the std::unique_ptr<mlx::core::array> are true (or non-zero).
        #[namespace = "mlx_cxx"]
        fn all(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        /// True if the two arrays are equal within the specified tolerance.
        #[namespace = "mlx_cxx"]
        fn allclose(
            a: &array,
            b: &array,
            rtol: f64,
            atol: f64,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "all_along_axes_keepdims"]
        fn all(
            a: &array,
            axes: &CxxVector<i32>,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "all_along_axis_keepdims"]
        fn all(a: &array, axis: i32, keepdims: bool, s: StreamOrDevice)
            -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "any_keepdims"]
        fn any(a: &array, keepdims: bool, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn any(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "any_along_axes_keepdims"]
        fn any(
            a: &array,
            axes: &CxxVector<i32>,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "any_along_axis_keepdims"]
        fn any(a: &array, axis: i32, keepdims: bool, s: StreamOrDevice)
            -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "sum_keepdims"]
        fn sum(a: &array, keepdims: bool, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn sum(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "sum_along_axes_keepdims"]
        fn sum(
            a: &array,
            axes: &CxxVector<i32>,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "sum_along_axis_keepdims"]
        fn sum(a: &array, axis: i32, keepdims: bool, s: StreamOrDevice)
            -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "mean_keepdims"]
        fn mean(a: &array, keepdims: bool, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn mean(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "mean_along_axes_keepdims"]
        fn mean(
            a: &array,
            axes: &CxxVector<i32>,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "mean_along_axis_keepdims"]
        fn mean(a: &array, axis: i32, keepdims: bool, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "var_keepdims"]
        fn var(a: &array, keepdims: bool, ddof: i32, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn var(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "var_along_axes_keepdims"]
        fn var(
            a: &array,
            axes: &CxxVector<i32>,
            keepdims: bool,
            ddof: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "var_along_axis_keepdims"]
        fn var(
            a: &array,
            axis: i32,
            keepdims: bool,
            ddof: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "prod_keepdims"]
        fn prod(a: &array, keepdims: bool, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn prod(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "prod_along_axes_keepdims"]
        fn prod(
            a: &array,
            axes: &CxxVector<i32>,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "prod_along_axis_keepdims"]
        fn prod(
            a: &array,
            axis: i32,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "max_keepdims"]
        fn max(a: &array, keepdims: bool, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn max(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "max_along_axes_keepdims"]
        fn max(
            a: &array,
            axes: &CxxVector<i32>,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "max_along_axis_keepdims"]
        fn max(a: &array, axis: i32, keepdims: bool, s: StreamOrDevice)
            -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "min_keepdims"]
        fn min(a: &array, keepdims: bool, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn min(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "min_along_axes_keepdims"]
        fn min(
            a: &array,
            axes: &CxxVector<i32>,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "min_along_axis_keepdims"]
        fn min(a: &array, axis: i32, keepdims: bool, s: StreamOrDevice)
            -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "argmin_keepdims"]
        fn argmin(a: &array, keepdims: bool, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn argmin(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "argmin_along_axis_keepdims"]
        fn argmin(
            a: &array,
            axis: i32,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "argmax_keepdims"]
        fn argmax(a: &array, keepdims: bool, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn argmax(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "argmax_along_axis_keepdims"]
        fn argmax(
            a: &array,
            axis: i32,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn sort(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "sort_along_axis"]
        fn sort(a: &array, axis: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn argsort(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "argsort_along_axis"]
        fn argsort(a: &array, axis: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn partition(a: &array, kth: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "partition_along_axis"]
        fn partition(a: &array, kth: i32, axis: i32, s: StreamOrDevice)
            -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn argpartition(a: &array, kth: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "argpartition_along_axis"]
        fn argpartition(
            a: &array,
            kth: i32,
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn topk(a: &array, k: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "topk_along_axis"]
        fn topk(a: &array, k: i32, axis: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "logsumexp_keepdims"]
        fn logsumexp(a: &array, keepdims: bool, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn logsumexp(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "logsumexp_along_axes_keepdims"]
        fn logsumexp(
            a: &array,
            axes: &CxxVector<i32>,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "logsumexp_along_axis_keepdims"]
        fn logsumexp(a: &array, axis: i32, keepdims: bool, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn abs(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn negative(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn sign(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn logical_not(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        // Returns a Result because it calls broadcast_arrays
        #[namespace = "mlx_cxx"]
        fn logical_and(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        // Returns a Result because it calls broadcast_arrays
        #[namespace = "mlx_cxx"]
        fn logical_or(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn reciprocal(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn add(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn subtract(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn multiply(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn divide(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn divmod(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<CxxVector<array>>>;

        #[namespace = "mlx_cxx"]
        fn floor_divide(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn remainder(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn maximum(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn minimum(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn floor(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn ceil(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn square(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn exp(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn sin(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn cos(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn tan(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn arcsin(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn arccos(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn arctan(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn sinh(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn cosh(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn tanh(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn arcsinh(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn arccosh(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn arctanh(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn log(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn log2(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn log10(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn log1p(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn logaddexp(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn sigmoid(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn erf(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn erfinv(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn stop_gradient(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "round_to_decimals"]
        fn round(a: &array, decimals: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn round(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn matmul(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "gather_along_axes"]
        fn gather(
            a: &array,
            indices: &[UniquePtr<array>],
            axes: &CxxVector<i32>,
            slice_sizes: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "gather_along_axis"]
        fn gather(
            a: &array,
            indices: &array,
            axis: i32,
            slice_sizes: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        /// Take std::unique_ptr<mlx::core::array> slices at the given indices of the specified axis.
        #[namespace = "mlx_cxx"]
        fn take(
            a: &array,
            indices: &array,
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        /// Take std::unique_ptr<mlx::core::array> entries at the given indices treating the
        /// std::unique_ptr<mlx::core::array> as flattened.
        #[namespace = "mlx_cxx"]
        #[rust_name = "take_flattened"]
        fn take(a: &array, indices: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn take_along_axis(
            a: &array,
            indices: &array,
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "scatter_along_axes"]
        fn scatter(
            a: &array,
            indices: &[UniquePtr<array>],
            updates: &array,
            axes: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "scatter_along_axis"]
        fn scatter(
            a: &array,
            indices: &array,
            updates: &array,
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "scatter_add_along_axes"]
        fn scatter_add(
            a: &array,
            indices: &[UniquePtr<array>],
            updates: &array,
            axes: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "scatter_add_along_axis"]
        fn scatter_add(
            a: &array,
            indices: &array,
            updates: &array,
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "scatter_prod_along_axes"]
        fn scatter_prod(
            a: &array,
            indices: &[UniquePtr<array>],
            updates: &array,
            axes: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "scatter_prod_along_axis"]
        fn scatter_prod(
            a: &array,
            indices: &array,
            updates: &array,
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "scatter_max_along_axes"]
        fn scatter_max(
            a: &array,
            indices: &[UniquePtr<array>],
            updates: &array,
            axes: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "scatter_max_along_axis"]
        fn scatter_max(
            a: &array,
            indices: &array,
            updates: &array,
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "scatter_min_along_axes"]
        fn scatter_min(
            a: &array,
            indices: &[UniquePtr<array>],
            updates: &array,
            axes: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "scatter_min_along_axis"]
        fn scatter_min(
            a: &array,
            indices: &array,
            updates: &array,
            axis: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn sqrt(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn rsqrt(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "softmax_along_axes"]
        fn softmax(a: &array, axes: &CxxVector<i32>, s: StreamOrDevice)
            -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn softmax(a: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "softmax_along_axis"]
        fn softmax(a: &array, axis: i32, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        /// Raise elements of a to the power of b element-wise
        #[namespace = "mlx_cxx"]
        fn power(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn cumsum(
            a: &array,
            axis: i32,
            reverse: bool,
            inclusive: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn cumprod(
            a: &array,
            axis: i32,
            reverse: bool,
            inclusive: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn cummax(
            a: &array,
            axis: i32,
            reverse: bool,
            inclusive: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn cummin(
            a: &array,
            axis: i32,
            reverse: bool,
            inclusive: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn conv1d(
            input: &array,
            weight: &array,
            stride: i32,
            padding: i32,
            dilation: i32,
            groups: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn conv2d(
            input: &array,
            weight: &array,
            stride: &[i32; 2],
            padding: &[i32; 2],
            dilation: &[i32; 2],
            groups: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn quantized_matmul(
            x: &array,
            w: &array,
            scales: &array,
            biases: &array,
            transpose: bool, // default: true
            group_size: i32, // default: 64
            bits: i32,       // default: 4
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn quantize(
            w: &array,
            group_size: i32, // default: 64
            bits: i32,       // default: 4
            s: StreamOrDevice,
        ) -> Result<[UniquePtr<array>; 3]>;

        #[namespace = "mlx_cxx"]
        fn dequantize(
            w: &array,
            scales: &array,
            biases: &array,
            group_size: i32, // default: 64
            bits: i32,       // default: 4
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "tensordot_ndims"]
        fn tensordot(
            a: &array,
            b: &array,
            dims: i32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "tensordot_list_dims"]
        fn tensordot(
            a: &array,
            b: &array,
            dims: &[UniquePtr<CxxVector<i32>>; 2],
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn outer(a: &array, b: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn inner(a: &array, b: &array, s: StreamOrDevice) -> Result<UniquePtr<array>>;

        /// Compute D = beta * C + alpha * (A @ B)
        #[namespace = "mlx_cxx"]
        fn addmm(
            c: UniquePtr<array>,
            a: UniquePtr<array>,
            b: UniquePtr<array>,
            alpha: &f32,
            beta: &f32,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::cxx_vec;

//     use super::ffi;

//     #[test]
//     fn test_arange() {
//         let array = ffi::arange_f64(0.0, 10.0, 1.0, Default::default());
//         assert_eq!(array.size(), 10);

//         let array = ffi::arange_i32(0, 10, 1, Default::default());
//         assert_eq!(array.size(), 10);
//     }

//     #[test]
//     fn test_arange_start_stop() {
//         let array = ffi::arange_start_stop_dtype(0.0, 10.0, crate::dtype::ffi::dtype_float32(), Default::default());
//         assert_eq!(array.size(), 10);

//         let array = ffi::arange_start_stop_dtype(0.0, 10.0, crate::dtype::ffi::dtype_int32(), Default::default());
//         assert_eq!(array.size(), 10);

//         let array = ffi::arange_start_stop_f64(0.0, 10.0, Default::default());
//         assert_eq!(array.size(), 10);

//         let array = ffi::arange_start_stop_i32(0, 10, Default::default());
//         assert_eq!(array.size(), 10);
//     }

//     #[test]
//     fn test_arange_stop() {
//         let array = ffi::arange_stop_dtype(10.0, crate::dtype::ffi::dtype_float32(), Default::default());
//         assert_eq!(array.size(), 10);

//         let array = ffi::arange_stop_dtype(10.0, crate::dtype::ffi::dtype_int32(), Default::default());
//         assert_eq!(array.size(), 10);

//         let array = ffi::arange_stop_f64(10.0, Default::default());
//         assert_eq!(array.size(), 10);

//         let array = ffi::arange_stop_i32(10, Default::default());
//         assert_eq!(array.size(), 10);
//     }

//     #[test]
//     fn test_linspace() {
//         let array = ffi::linspace(0.0, 10.0, 10, crate::dtype::ffi::dtype_float32(), Default::default());
//         assert_eq!(array.size(), 10);
//     }

//     #[test]
//     fn test_astype() {
//         let array = ffi::arange_f64(0.0, 10.0, 1.0, Default::default());
//         let array = ffi::astype(&array, crate::dtype::ffi::dtype_int32(), Default::default());
//         assert_eq!(array.size(), 10);
//     }

//     #[test]
//     fn test_as_strided() {
//         let array = ffi::arange_f64(0.0, 10.0, 1.0, Default::default());
//         let array = ffi::as_strided(
//             &array,
//             cxx_vec![2, 5],
//             cxx_vec![1, 5], // TODO: Is this correct?
//             0,
//             Default::default(),
//         );
//         // TODO: how to check
//     }
// }
