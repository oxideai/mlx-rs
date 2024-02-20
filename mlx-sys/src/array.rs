use crate::dtype::ffi::Dtype;

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/array.h");
        include!("mlx-cxx/mlx_cxx.hpp");
        include!("mlx-cxx/array.hpp");

        #[namespace = "mlx::core"]
        type float16_t = crate::types::float16::float16_t;

        #[namespace = "mlx::core"]
        type bfloat16_t = crate::types::bfloat16::bfloat16_t;

        #[namespace = "mlx::core"]
        type complex64_t = crate::types::complex64::complex64_t;

        #[namespace = "mlx::core"]
        type array;

        // TODO: is uintptr_t always usize?
        #[cxx_name = "size_t"]
        type uintptr_t = libc::uintptr_t;

        #[namespace = "mlx_cxx"]
        fn array_empty(dtype: Dtype) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_bool(value: bool) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_int8(value: i8) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_int16(value: i16) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_int32(value: i32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_int64(value: i64) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_uint8(value: u8) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_uint16(value: u16) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_uint32(value: u32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_uint64(value: u64) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_float32(value: f32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_float16(value: float16_t) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_bfloat16(value: bfloat16_t) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_complex64(value: complex64_t) -> UniquePtr<array>;

        #[namespace = "mlx::core"]
        fn itemsize(self: &array) -> usize;

        #[namespace = "mlx::core"]
        fn size(self: &array) -> usize;

        #[namespace = "mlx::core"]
        fn nbytes(self: &array) -> usize;

        #[namespace = "mlx::core"]
        fn ndim(self: &array) -> usize;

        #[namespace = "mlx::core"]
        #[rust_name = "shape"]
        fn shape(self: &array) -> &CxxVector<i32>;

        #[namespace = "mlx::core"]
        #[rust_name = "shape_of_dim"]
        fn shape(self: &array, dim: i32) -> i32;

        #[namespace = "mlx::core"]
        fn strides(self: &array) -> &CxxVector<usize>;

        #[namespace = "mlx::core"]
        type Dtype = crate::dtype::ffi::Dtype;

        #[namespace = "mlx::core"]
        fn dtype(self: &array) -> Dtype;

        #[namespace = "mlx::core"]
        fn eval(self: Pin<&mut array>) -> Result<()>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_bool(self: Pin<&mut array>) -> Result<bool>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_uint8(self: Pin<&mut array>) -> Result<u8>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_uint16(self: Pin<&mut array>) -> Result<u16>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_uint32(self: Pin<&mut array>) -> Result<u32>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_uint64(self: Pin<&mut array>) -> Result<u64>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_int8(self: Pin<&mut array>) -> Result<i8>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_int16(self: Pin<&mut array>) -> Result<i16>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_int32(self: Pin<&mut array>) -> Result<i32>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_int64(self: Pin<&mut array>) -> Result<i64>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_float16(self: Pin<&mut array>) -> Result<float16_t>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_bfloat16(self: Pin<&mut array>) -> Result<bfloat16_t>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_float32(self: Pin<&mut array>) -> Result<f32>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_mut_complex64(self: Pin<&mut array>) -> Result<complex64_t>;


        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_bool(self: &array) -> Result<bool>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_uint8(self: &array) -> Result<u8>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_uint16(self: &array) -> Result<u16>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_uint32(self: &array) -> Result<u32>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_uint64(self: &array) -> Result<u64>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_int8(self: &array) -> Result<i8>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_int16(self: &array) -> Result<i16>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_int32(self: &array) -> Result<i32>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_int64(self: &array) -> Result<i64>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_float16(self: &array) -> Result<float16_t>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_bfloat16(self: &array) -> Result<bfloat16_t>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_float32(self: &array) -> Result<f32>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_complex64(self: &array) -> Result<complex64_t>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_bool(slice: &[bool], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_uint8(slice: &[u8], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_uint16(slice: &[u16], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_uint32(slice: &[u32], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_uint64(slice: &[u64], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_int8(slice: &[i8], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_int16(slice: &[i16], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_int32(slice: &[i32], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_int64(slice: &[i64], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_float16(
            slice: &[float16_t],
            shape: &CxxVector<i32>,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_bfloat16(
            slice: &[bfloat16_t],
            shape: &CxxVector<i32>,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_float32(slice: &[f32], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_complex64(
            slice: &[complex64_t],
            shape: &CxxVector<i32>,
        ) -> UniquePtr<array>;

        #[namespace = "mlx::core"]
        fn id(self: &array) -> uintptr_t;

        #[namespace = "mlx::core"]
        fn primitive_id(self: &array) -> uintptr_t;

        // TODO: should method `primitive()` be exposed?

        #[namespace = "mlx::core"]
        fn has_primitive(self: &array) -> bool;

        #[namespace = "mlx::core"]
        fn inputs(self: &array) -> &CxxVector<array>;

        #[namespace = "mlx::core"]
        fn is_donatable(self: &array) -> bool;

        #[namespace = "mlx::core"]
        fn siblings(self: &array) -> &CxxVector<array>;

        // TODO: user cannot create `CxxVector<array>` directly.
        #[namespace = "mlx_cxx"]
        fn set_array_siblings(
            arr: Pin<&mut array>,
            siblings: UniquePtr<CxxVector<array>>,
            position: u16,
        );

        #[namespace = "mlx_cxx"]
        fn array_outputs(arr: Pin<&mut array>) -> UniquePtr<CxxVector<array>>;

        // TODO: expose Flags

        #[namespace = "mlx::core"]
        fn graph_depth(self: &array) -> u16;

        #[namespace = "mlx::core"]
        fn detach(self: Pin<&mut array>);

        #[namespace = "mlx::core"]
        fn data_size(self: &array) -> usize;

        // TODO: expose allocator::Buffer?

        // TODO: expose array::Data and data_shared_ptr()?

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_bool(self: Pin<&mut array>) -> *mut bool;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_uint8(self: Pin<&mut array>) -> *mut u8;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_uint16(self: Pin<&mut array>) -> *mut u16;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_uint32(self: Pin<&mut array>) -> *mut u32;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_uint64(self: Pin<&mut array>) -> *mut u64;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_int8(self: Pin<&mut array>) -> *mut i8;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_int16(self: Pin<&mut array>) -> *mut i16;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_int32(self: Pin<&mut array>) -> *mut i32;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_int64(self: Pin<&mut array>) -> *mut i64;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_float16(self: Pin<&mut array>) -> *mut float16_t;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_bfloat16(self: Pin<&mut array>) -> *mut bfloat16_t;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_float32(self: Pin<&mut array>) -> *mut f32;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_mut_complex64(self: Pin<&mut array>) -> *mut complex64_t;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_bool(self: &array) -> *const bool;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_uint8(self: &array) -> *const u8;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_uint16(self: &array) -> *const u16;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_uint32(self: &array) -> *const u32;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_uint64(self: &array) -> *const u64;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_int8(self: &array) -> *const i8;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_int16(self: &array) -> *const i16;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_int32(self: &array) -> *const i32;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_int64(self: &array) -> *const i64;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_float16(self: &array) -> *const float16_t;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_bfloat16(self: &array) -> *const bfloat16_t;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_float32(self: &array) -> *const f32;

        #[namespace = "mlx::core"]
        #[cxx_name = "data"]
        fn data_complex64(self: &array) -> *const complex64_t;

        #[namespace = "mlx::core"]
        fn is_evaled(self: &array) -> bool;

        #[namespace = "mlx::core"]
        fn set_tracer(self: Pin<&mut array>, is_tracer: bool);

        #[namespace = "mlx::core"]
        fn is_tracer(self: &array) -> bool;

        // TODO: should these method be exposed?
        // 1. `set_data()`,
        // 2. `copy_shared_buffer()`
        // 3. `move_shared_buffer()`

        #[namespace = "mlx::core"]
        fn overwrite_descriptor(self: Pin<&mut array>, other: &array);
    }

    impl CxxVector<array> {} // Explicit instantiation
}
