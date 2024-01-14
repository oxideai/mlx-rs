use cxx::UniquePtr;

use crate::{Optional, array::ffi::array};

type OptionalArray = Optional<UniquePtr<array>>;

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
        fn arange(
            start: f64,
            stop: f64,
            step: f64,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_start_stop_dtype"]
        fn arange(
            start: f64,
            stop: f64,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_start_stop_f64"]
        fn arange(start: f64, stop: f64, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_stop_dtype"]
        fn arange(stop: f64, dtype: Dtype, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_stop_f64"]
        fn arange(stop: f64, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_i32"]
        fn arange(
            start: i32,
            stop: i32,
            step: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_start_stop_i32"]
        fn arange(start: i32, stop: i32, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "arange_stop_i32"]
        fn arange(stop: i32, s: StreamOrDevice) -> UniquePtr<array>;

        // A 1D std::unique_ptr<mlx::core::array> of `num` evenly spaced numbers in the range
        // `[start, stop]`
        #[namespace = "mlx_cxx"]
        fn linspace(
            start: f64,
            stop: f64,
            num: i32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        // Convert an array to the given data type.
        #[namespace = "mlx_cxx"]
        fn astype(
            a: &array,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        // Create a view of an array with the given shape and strides.
        #[namespace = "mlx_cxx"]
        fn as_strided(
            a: &array,
            shape: UniquePtr<CxxVector<i32>>,
            strides: UniquePtr<CxxVector<usize>>,
            offset: usize,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn copy(a: &array, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "full_vals_dtype"]
        fn full(
            shape: &CxxVector<i32>,
            vals: &array,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "full_vals"]
        fn full(
            shape: &CxxVector<i32>,
            vals: &array,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_bool_val_dtype(
            shape: &CxxVector<i32>,
            val: bool,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_uint8_val_dtype(
            shape: &CxxVector<i32>,
            val: u8,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_uint16_val_dtype(
            shape: &CxxVector<i32>,
            val: u16,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_uint32_val_dtype(
            shape: &CxxVector<i32>,
            val: u32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_uint64_val_dtype(
            shape: &CxxVector<i32>,
            val: u64,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_int8_val_dtype(
            shape: &CxxVector<i32>,
            val: i8,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_int16_val_dtype(
            shape: &CxxVector<i32>,
            val: i16,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_int32_val_dtype(
            shape: &CxxVector<i32>,
            val: i32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_int64_val_dtype(
            shape: &CxxVector<i32>,
            val: i64,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_float16_val_dtype(
            shape: &CxxVector<i32>,
            val: float16_t,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_bfloat16_val_dtype(
            shape: &CxxVector<i32>,
            val: bfloat16_t,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_float32_val_dtype(
            shape: &CxxVector<i32>,
            val: f32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_complex64_val_dtype(
            shape: &CxxVector<i32>,
            val: complex64_t,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_bool_val(
            shape: &CxxVector<i32>,
            val: bool,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_uint8_val(
            shape: &CxxVector<i32>,
            val: u8,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_uint16_val(
            shape: &CxxVector<i32>,
            val: u16,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_uint32_val(
            shape: &CxxVector<i32>,
            val: u32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_uint64_val(
            shape: &CxxVector<i32>,
            val: u64,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_int8_val(
            shape: &CxxVector<i32>,
            val: i8,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_int16_val(
            shape: &CxxVector<i32>,
            val: i16,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_int32_val(
            shape: &CxxVector<i32>,
            val: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_int64_val(
            shape: &CxxVector<i32>,
            val: i64,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_float16_val(
            shape: &CxxVector<i32>,
            val: float16_t,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_bfloat16_val(
            shape: &CxxVector<i32>,
            val: bfloat16_t,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_float32_val(
            shape: &CxxVector<i32>,
            val: f32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn full_complex64_val(
            shape: &CxxVector<i32>,
            val: complex64_t,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "zeros_dtype"]
        fn zeros(
            shape: &CxxVector<i32>,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn zeros(shape: &CxxVector<i32>, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn zeros_like(
            a: &array,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "ones_dtype"]
        fn ones(
            shape: &CxxVector<i32>,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn ones(shape: &CxxVector<i32>, s: StreamOrDevice) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn ones_like(
            a: &array,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "eye_n_m_k_dtype"]
        fn eye(
            n: i32,
            m: i32,
            k: i32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "eye_n_dtype"]
        fn eye(
            n: i32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "eye_n_m"]
        fn eye(
            n: i32,
            m: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "eye_n_m_k"]
        fn eye(
            n: i32,
            m: i32,
            k: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "eye_n"]
        fn eye(
            n: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "identity_dtype"]
        fn identity(
            n: i32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn identity(n: i32, s: StreamOrDevice) -> UniquePtr<array>;
        
        #[namespace = "mlx_cxx"]
        #[rust_name = "tri_n_m_k_dtype"]
        fn tri(
            n: i32,
            m: i32,
            k: i32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "tri_n_dtype"]
        fn tri(
            n: i32,
            dtype: Dtype,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn tril(
            x: UniquePtr<array>,
            k: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn triu(
            x: UniquePtr<array>,
            k: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn reshape(
            a: &array,
            shape: UniquePtr<CxxVector<i32>>,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "flatten_start_axis_end_axis"]
        fn flatten(
            a: &array,
            start_axis: i32,
            end_axis: i32,
            s : StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn flatten(
            a: &array,
            s : StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "squeeze_axes"]
        fn squeeze(
            a: &array,
            axes: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "squeeze_axis"]
        fn squeeze(
            a: &array,
            axis: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn squeeze(
            a: &array,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "expand_dims_at_axes"]
        fn expand_dims(
            a: &array,
            axes: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "expand_dims_at_axis"]
        fn expand_dims(
            a: &array,
            axis: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "slice_start_stop_strides"]
        fn slice(
            a: &array,
            start: UniquePtr<CxxVector<i32>>,
            stop: UniquePtr<CxxVector<i32>>,
            strides: UniquePtr<CxxVector<i32>>,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn slice(
            a: &array,
            start: &CxxVector<i32>,
            stop: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "split_n_at_axis"]
        fn split(
            a: &array,
            num_splits: i32,
            axis: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<CxxVector<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "split_n"]
        fn split(
            a: &array,
            num_splits: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<CxxVector<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "split_at_indices_along_axis"]
        fn split(
            a: &array,
            indices: &CxxVector<i32>,
            axis: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<CxxVector<array>>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "split_at_indices"]
        fn split(
            a: &array,
            indices: &CxxVector<i32>,
            s: StreamOrDevice,
        ) -> UniquePtr<CxxVector<array>>;

        #[namespace = "mlx_cxx"]
        fn clip(
            a: &array,
            a_min: &OptionalArray,
            a_max: &OptionalArray,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "concatenate_along_axis"]
        fn concatenate(
            arrays: &[UniquePtr<array>],
            axis: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn concatenate(
            arrays: &[UniquePtr<array>],
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "stack_along_axis"]
        fn stack(
            arrays: &[UniquePtr<array>],
            axis: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn stack(
            arrays: &[UniquePtr<array>],
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "repeat_along_axis"]
        fn repeat(
            arr: &array,
            repeats: i32,
            axis: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn repeat(
            arr: &array,
            repeats: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn transpose(
            a: &array,
            axes: UniquePtr<CxxVector<i32>>,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;
    
        #[namespace = "mlx_cxx"]
        fn swapaxes(
            a: &array,
            axis1: i32,
            axis2: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn moveaxis(
            a: &array,
            source: i32,
            destination: i32,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "pad_axes"]
        fn pad(
            a: &array,
            axes: &CxxVector<i32>,
            low_pad_size: &CxxVector<i32>,
            high_pad_size: &CxxVector<i32>,
            pad_value: &array,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "pad_unique_widths_for_each_axis"]
        fn pad(
            a: &array,
            pad_width: &[[i32; 2]],
            pad_value: &array,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[rust_name = "pad_same_widths_for_each_axis"]
        fn pad(
            a: &array,
            pad_width: &[i32; 2],
            pad_value: &array,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn pad(
            a: &array,
            pad_width: i32,
            pad_value: &array,
            s: StreamOrDevice,
        ) -> UniquePtr<array>;
    }
}

#[cfg(test)]
mod tests {
    use crate::cxx_vec;

    use super::ffi;

    #[test]
    fn test_arange() {
        let array = ffi::arange_f64(0.0, 10.0, 1.0, Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_i32(0, 10, 1, Default::default());
        assert_eq!(array.size(), 10);
    }

    #[test]
    fn test_arange_start_stop() {
        let array = ffi::arange_start_stop_dtype(0.0, 10.0, crate::dtype::ffi::dtype_float32(), Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_start_stop_dtype(0.0, 10.0, crate::dtype::ffi::dtype_int32(), Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_start_stop_f64(0.0, 10.0, Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_start_stop_i32(0, 10, Default::default());
        assert_eq!(array.size(), 10);
    }

    #[test]
    fn test_arange_stop() {
        let array = ffi::arange_stop_dtype(10.0, crate::dtype::ffi::dtype_float32(), Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_stop_dtype(10.0, crate::dtype::ffi::dtype_int32(), Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_stop_f64(10.0, Default::default());
        assert_eq!(array.size(), 10);

        let array = ffi::arange_stop_i32(10, Default::default());
        assert_eq!(array.size(), 10);
    }

    #[test]
    fn test_linspace() {
        let array = ffi::linspace(0.0, 10.0, 10, crate::dtype::ffi::dtype_float32(), Default::default());
        assert_eq!(array.size(), 10);
    }

    #[test]
    fn test_astype() {
        let array = ffi::arange_f64(0.0, 10.0, 1.0, Default::default());
        let array = ffi::astype(&array, crate::dtype::ffi::dtype_int32(), Default::default());
        assert_eq!(array.size(), 10);
    }

    #[test]
    fn test_as_strided() {
        let array = ffi::arange_f64(0.0, 10.0, 1.0, Default::default());
        let array = ffi::as_strided(
            &array,
            cxx_vec![2, 5],
            cxx_vec![1, 5], // TODO: Is this correct?
            0,
            Default::default(),
        );
        // TODO: how to check
    }
}
