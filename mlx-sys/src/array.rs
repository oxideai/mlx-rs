//! TODO: add bindings to constructors that takes `Primitives`?

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
        #[cxx_name = "new_unique"]
        fn array_new_bool(value: bool) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_i8(value: i8) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_i16(value: i16) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_i32(value: i32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_i64(value: i64) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_u8(value: u8) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_u16(value: u16) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_u32(value: u32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_u64(value: u64) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_f32(value: f32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_f16(value: float16_t) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_bf16(value: bfloat16_t) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_c64(value: complex64_t) -> UniquePtr<array>;

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
        fn eval(self: Pin<&mut array>);

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_bool(self: Pin<&mut array>) -> Result<bool>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_uint8(self: Pin<&mut array>) -> Result<u8>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_uint16(self: Pin<&mut array>) -> Result<u16>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_uint32(self: Pin<&mut array>) -> Result<u32>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_uint64(self: Pin<&mut array>) -> Result<u64>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_int8(self: Pin<&mut array>) -> Result<i8>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_int16(self: Pin<&mut array>) -> Result<i16>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_int32(self: Pin<&mut array>) -> Result<i32>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_int64(self: Pin<&mut array>) -> Result<i64>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_float16(self: Pin<&mut array>) -> Result<float16_t>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_bfloat16(self: Pin<&mut array>) -> Result<bfloat16_t>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_float32(self: Pin<&mut array>) -> Result<f32>;

        #[namespace = "mlx::core"]
        #[cxx_name = "item"]
        fn item_complex64(self: Pin<&mut array>) -> Result<complex64_t>;

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

// #[cfg(test)]
// mod tests {
//     use crate::cxx_vec;

//     use super::*;

//     #[test]
//     fn test_array_new_bool() {
//         let mut array = ffi::array_new_bool(true);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::bool_));

//         let item = array.pin_mut().item_bool();
//         assert_eq!(item, true);
//     }

//     #[test]
//     fn test_array_new_i8() {
//         let mut array = ffi::array_new_i8(1);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::int8));

//         let item = array.pin_mut().item_int8();
//         assert_eq!(item, 1);
//     }

//     #[test]
//     fn test_array_new_i16() {
//         let mut array = ffi::array_new_i16(1);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::int16));

//         let item = array.pin_mut().item_int16();
//         assert_eq!(item, 1);
//     }

//     #[test]
//     fn test_array_new_i32() {
//         let mut array = ffi::array_new_i32(1);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::int32));

//         let item = array.pin_mut().item_int32();
//         assert_eq!(item, 1);
//     }

//     #[test]
//     fn test_array_new_i64() {
//         let mut array = ffi::array_new_i64(1);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::int64));

//         let item = array.pin_mut().item_int64();
//         assert_eq!(item, 1);
//     }

//     #[test]
//     fn test_array_new_u8() {
//         let mut array = ffi::array_new_u8(1);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint8));

//         let item = array.pin_mut().item_uint8();
//         assert_eq!(item, 1);
//     }

//     #[test]
//     fn test_array_new_u16() {
//         let mut array = ffi::array_new_u16(1);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint16));

//         let item = array.pin_mut().item_uint16();
//         assert_eq!(item, 1);
//     }

//     #[test]
//     fn test_array_new_u32() {
//         let mut array = ffi::array_new_u32(1);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint32));

//         let item = array.pin_mut().item_uint32();
//         assert_eq!(item, 1);
//     }

//     #[test]
//     fn test_array_new_u64() {
//         let mut array = ffi::array_new_u64(1);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint64));

//         let item = array.pin_mut().item_uint64();
//         assert_eq!(item, 1);
//     }

//     #[test]
//     fn test_array_new_f32() {
//         let mut array = ffi::array_new_f32(1.0);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::float32));

//         let item = array.pin_mut().item_float32();
//         assert_eq!(item, 1.0);
//     }

//     #[test]
//     fn test_array_new_f16() {
//         let mut array = ffi::array_new_f16(ffi::float16_t { bits: 0x3c00 });
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::float16));

//         let item = array.pin_mut().item_float16();
//         assert_eq!(item.bits, 0x3c00);
//     }

//     #[test]
//     fn test_array_new_bf16() {
//         let mut array = ffi::array_new_bf16(ffi::bfloat16_t { bits: 0x3c00 });
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::bfloat16));

//         let item = array.pin_mut().item_bfloat16();
//         assert_eq!(item.bits, 0x3c00);
//     }

//     #[test]
//     fn test_array_new_c64() {
//         let mut array = ffi::array_new_c64(ffi::complex64_t { re: 1.0, im: 1.0 });
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 1);

//         let dtype = array.dtype();
//         assert!(matches!(
//             dtype.val,
//             crate::dtype::ffi::Val::complex64
//         ));

//         let item = array.pin_mut().item_complex64();
//         assert_eq!(item.re, 1.0);
//         assert_eq!(item.im, 1.0);
//     }

//     #[test]
//     fn test_array_from_slice_bool() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_bool(&[true, false], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::bool_));
//     }

//     #[test]
//     fn test_array_from_slice_uint8() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_uint8(&[1, 2], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint8));
//     }

//     #[test]
//     fn test_array_from_slice_uint16() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_uint16(&[1, 2], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint16));
//     }

//     #[test]
//     fn test_array_from_slice_uint32() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_uint32(&[1, 2], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint32));
//     }

//     #[test]
//     fn test_array_from_slice_uint64() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_uint64(&[1, 2], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint64));
//     }

//     #[test]
//     fn test_array_from_slice_int8() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_int8(&[1, 2], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::int8));
//     }

//     #[test]
//     fn test_array_from_slice_int16() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_int16(&[1, 2], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::int16));
//     }

//     #[test]
//     fn test_array_from_slice_int32() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_int32(&[1, 2], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::int32));
//     }

//     #[test]
//     fn test_array_from_slice_int64() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_int64(&[1, 2], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::int64));
//     }

//     #[test]
//     fn test_array_from_slice_float16() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_float16(&[ffi::float16_t { bits: 0x3c00 }, ffi::float16_t { bits: 0x3c00 }], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(
//             dtype.val,
//             crate::dtype::ffi::Val::float16
//         ));
//     }

//     #[test]
//     fn test_array_from_slice_bfloat16() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_bfloat16(&[ffi::bfloat16_t { bits: 0x3c00 }, ffi::bfloat16_t { bits: 0x3c00 }], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(
//             dtype.val,
//             crate::dtype::ffi::Val::bfloat16
//         ));
//     }

//     #[test]
//     fn test_array_from_slice_float32() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_float32(&[1.0, 2.0], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(
//             dtype.val,
//             crate::dtype::ffi::Val::float32
//         ));
//     }

//     #[test]
//     fn test_array_from_slice_complex64() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_complex64(&[ffi::complex64_t { re: 1.0, im: 1.0 }, ffi::complex64_t { re: 1.0, im: 1.0 }], &shape);
//         assert!(!array.is_null());
//         assert_eq!(array.size(), 2);

//         let dtype = array.dtype();
//         assert!(matches!(
//             dtype.val,
//             crate::dtype::ffi::Val::complex64
//         ));
//     }

//     #[test]
//     fn test_array_itemsize() {
//         let array = ffi::array_new_bool(true);
//         assert_eq!(array.itemsize(), 1);
//     }

//     #[test]
//     fn test_array_size() {
//         let array = ffi::array_new_bool(true);
//         assert_eq!(array.size(), 1);
//     }

//     #[test]
//     fn test_array_nbytes() {
//         let array = ffi::array_new_bool(true);
//         assert_eq!(array.nbytes(), 1);
//     }

//     #[test]
//     fn test_array_ndim() {
//         let array = ffi::array_new_bool(true);
//         let _ndim = array.ndim();
//     }

//     #[test]
//     fn test_array_shape() {
//         let array = ffi::array_new_bool(true);
//         let _shape = array.shape();
//     }

//     #[test]
//     fn test_array_strides() {
//         let array = ffi::array_new_bool(true);
//         let _strides = array.strides();
//     }

//     #[test]
//     fn test_array_dtype() {
//         let array = ffi::array_new_bool(true);
//         let dtype = array.dtype();
//         assert!(matches!(dtype.val, crate::dtype::ffi::Val::bool_));
//     }

//     #[test]
//     fn test_array_eval() {
//         let a = ffi::array_new_f32(1.0);
//         let b = ffi::array_new_f32(2.0);
//         let mut c = crate::ops::ffi::add(&a, &b, Default::default());
//         c.pin_mut().eval();
//         assert_eq!(c.pin_mut().item_float32(), 3.0);
//     }

//     #[test]
//     fn test_array_id() {
//         let array = ffi::array_new_bool(true);
//         let _id = array.id();
//     }

//     #[test]
//     fn test_array_primitive_id() {
//         let array = ffi::array_new_bool(true);
//         let _id = array.primitive_id();
//     }

//     #[test]
//     fn test_array_has_primitive() {
//         let array = ffi::array_new_bool(true);
//         let _has_primitive = array.has_primitive();
//     }

//     #[test]
//     fn test_array_inputs() {
//         let array = ffi::array_new_bool(true);
//         let _inputs = array.inputs();
//     }

//     #[test]
//     fn test_array_siblings() {
//         let array = ffi::array_new_bool(true);
//         let _siblings = array.siblings();
//     }

//     #[test]
//     fn test_array_set_siblings() {
//         let mut array = ffi::array_new_bool(true);
//         let siblings = cxx::CxxVector::new();
//         ffi::set_array_siblings(array.pin_mut(), siblings, 0);
//     }

//     #[test]
//     fn test_array_outputs() {
//         let mut array = ffi::array_new_bool(true);
//         let _outputs = ffi::array_outputs(array.pin_mut());
//     }

//     #[test]
//     fn test_array_detach() {
//         let mut array = ffi::array_new_bool(true);
//         array.pin_mut().detach();
//     }

//     #[test]
//     fn test_array_data_size() {
//         let array = ffi::array_new_bool(true);
//         let _data_size = array.data_size();
//     }

//     #[test]
//     fn test_array_data_mut_bool() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_bool(&[true, false], &shape);
//         let ptr = array.pin_mut().data_mut_bool();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as bool, true);
//             assert_eq!(*ptr.offset(1) as bool, false);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_uint8() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_uint8(&[1, 2], &shape);
//         let ptr = array.pin_mut().data_mut_uint8();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as u8, 1);
//             assert_eq!(*ptr.offset(1) as u8, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_uint16() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_uint16(&[1, 2], &shape);
//         let ptr = array.pin_mut().data_mut_uint16();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as u16, 1);
//             assert_eq!(*ptr.offset(1) as u16, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_uint32() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_uint32(&[1, 2], &shape);
//         let ptr = array.pin_mut().data_mut_uint32();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as u32, 1);
//             assert_eq!(*ptr.offset(1) as u32, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_uint64() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_uint64(&[1, 2], &shape);
//         let ptr = array.pin_mut().data_mut_uint64();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as u64, 1);
//             assert_eq!(*ptr.offset(1) as u64, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_int8() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_int8(&[1, 2], &shape);
//         let ptr = array.pin_mut().data_mut_int8();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as i8, 1);
//             assert_eq!(*ptr.offset(1) as i8, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_int16() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_int16(&[1, 2], &shape);
//         let ptr = array.pin_mut().data_mut_int16();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as i16, 1);
//             assert_eq!(*ptr.offset(1) as i16, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_int32() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_int32(&[1, 2], &shape);
//         let ptr = array.pin_mut().data_mut_int32();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as i32, 1);
//             assert_eq!(*ptr.offset(1) as i32, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_int64() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_int64(&[1, 2], &shape);
//         let ptr = array.pin_mut().data_mut_int64();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as i64, 1);
//             assert_eq!(*ptr.offset(1) as i64, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_float16() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_float16(&[ffi::float16_t { bits: 0x3c00 }, ffi::float16_t { bits: 0x3c00 }], &shape);
//         let ptr = array.pin_mut().data_mut_float16();
//         unsafe {
//             assert_eq!((*ptr.offset(0)).bits, 0x3c00);
//             assert_eq!((*ptr.offset(1)).bits, 0x3c00);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_bfloat16() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_bfloat16(&[ffi::bfloat16_t { bits: 0x3c00 }, ffi::bfloat16_t { bits: 0x3c00 }], &shape);
//         let ptr = array.pin_mut().data_mut_bfloat16();
//         unsafe {
//             assert_eq!((*ptr.offset(0)).bits, 0x3c00);
//             assert_eq!((*ptr.offset(1)).bits, 0x3c00);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_float32() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_float32(&[1.0, 2.0], &shape);
//         let ptr = array.pin_mut().data_mut_float32();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as f32, 1.0);
//             assert_eq!(*ptr.offset(1) as f32, 2.0);
//         }
//     }

//     #[test]
//     fn test_array_data_mut_complex64() {
//         let shape = cxx_vec![2];
//         let mut array = ffi::array_from_slice_complex64(&[ffi::complex64_t { re: 1.0, im: 1.0 }, ffi::complex64_t { re: 1.0, im: 1.0 }], &shape);
//         let ptr = array.pin_mut().data_mut_complex64();
//         unsafe {
//             assert_eq!((*ptr.offset(0)).re, 1.0);
//             assert_eq!((*ptr.offset(0)).im, 1.0);
//             assert_eq!((*ptr.offset(1)).re, 1.0);
//             assert_eq!((*ptr.offset(1)).im, 1.0);
//         }
//     }

//     #[test]
//     fn test_array_data_bool() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_bool(&[true, false], &shape);
//         let ptr = array.data_bool();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as bool, true);
//             assert_eq!(*ptr.offset(1) as bool, false);
//         }
//     }

//     #[test]
//     fn test_array_data_uint8() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_uint8(&[1, 2], &shape);
//         let ptr = array.data_uint8();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as u8, 1);
//             assert_eq!(*ptr.offset(1) as u8, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_uint16() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_uint16(&[1, 2], &shape);
//         let ptr = array.data_uint16();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as u16, 1);
//             assert_eq!(*ptr.offset(1) as u16, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_uint32() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_uint32(&[1, 2], &shape);
//         let ptr = array.data_uint32();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as u32, 1);
//             assert_eq!(*ptr.offset(1) as u32, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_uint64() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_uint64(&[1, 2], &shape);
//         let ptr = array.data_uint64();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as u64, 1);
//             assert_eq!(*ptr.offset(1) as u64, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_int8() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_int8(&[1, 2], &shape);
//         let ptr = array.data_int8();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as i8, 1);
//             assert_eq!(*ptr.offset(1) as i8, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_int16() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_int16(&[1, 2], &shape);
//         let ptr = array.data_int16();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as i16, 1);
//             assert_eq!(*ptr.offset(1) as i16, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_int32() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_int32(&[1, 2], &shape);
//         let ptr = array.data_int32();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as i32, 1);
//             assert_eq!(*ptr.offset(1) as i32, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_int64() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_int64(&[1, 2], &shape);
//         let ptr = array.data_int64();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as i64, 1);
//             assert_eq!(*ptr.offset(1) as i64, 2);
//         }
//     }

//     #[test]
//     fn test_array_data_float16() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_float16(&[ffi::float16_t { bits: 0x3c00 }, ffi::float16_t { bits: 0x3c00 }], &shape);
//         let ptr = array.data_float16();
//         unsafe {
//             assert_eq!((*ptr.offset(0)).bits, 0x3c00);
//             assert_eq!((*ptr.offset(1)).bits, 0x3c00);
//         }
//     }

//     #[test]
//     fn test_array_data_bfloat16() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_bfloat16(&[ffi::bfloat16_t { bits: 0x3c00 }, ffi::bfloat16_t { bits: 0x3c00 }], &shape);
//         let ptr = array.data_bfloat16();
//         unsafe {
//             assert_eq!((*ptr.offset(0)).bits, 0x3c00);
//             assert_eq!((*ptr.offset(1)).bits, 0x3c00);
//         }
//     }

//     #[test]
//     fn test_array_data_float32() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_float32(&[1.0, 2.0], &shape);
//         let ptr = array.data_float32();
//         unsafe {
//             assert_eq!(*ptr.offset(0) as f32, 1.0);
//             assert_eq!(*ptr.offset(1) as f32, 2.0);
//         }
//     }

//     #[test]
//     fn test_array_data_complex64() {
//         let shape = cxx_vec![2];
//         let array = ffi::array_from_slice_complex64(&[ffi::complex64_t { re: 1.0, im: 1.0 }, ffi::complex64_t { re: 1.0, im: 1.0 }], &shape);
//         let ptr = array.data_complex64();
//         unsafe {
//             assert_eq!((*ptr.offset(0)).re, 1.0);
//             assert_eq!((*ptr.offset(0)).im, 1.0);
//             assert_eq!((*ptr.offset(1)).re, 1.0);
//             assert_eq!((*ptr.offset(1)).im, 1.0);
//         }
//     }

//     #[test]
//     fn test_array_is_evaled() {
//         let array = ffi::array_new_bool(true);
//         let _is_evaled = array.is_evaled();
//     }

//     #[test]
//     fn test_array_set_tracer() {
//         let mut array = ffi::array_new_bool(true);
//         array.pin_mut().set_tracer(true);
//     }

//     #[test]
//     fn test_array_is_tracer() {
//         let array = ffi::array_new_bool(true);
//         let _is_tracer = array.is_tracer();
//     }

//     #[test]
//     fn test_array_overwrite_descriptor() {
//         let mut a = ffi::array_new_bool(true);
//         let b = ffi::array_new_bool(true);
//         a.pin_mut().overwrite_descriptor(&b);
//     }
// }
