use half::{bf16, f16};
use mlx_sys::{__BindgenComplex, bfloat16_t, float16_t, mlx_array};

use crate::{complex64, error::{setup_mlx_error_handler, Exception}, Array};

use super::SUCCESS;

type Status = i32;

pub trait Guard<T>: Default {
    type MutRawPtr;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr;

    fn set_init_success(&mut self, success: bool);

    fn try_into_guarded(self) -> Result<T, Exception>;
}

pub(crate) trait Guarded: Sized {
    type Guard: Guard<Self>;

    fn try_op<F>(f: F) -> Result<Self, Exception>
    where 
        F: FnOnce(<Self::Guard as Guard<Self>>::MutRawPtr) -> Status,
    {
        crate::error::INIT_ERR_HANDLER.with(|init| {
            init.call_once(setup_mlx_error_handler)
        });

        let mut guard = Self::Guard::default();
        let status = f(guard.as_mut_raw_ptr());
        match status {
            SUCCESS => {
                guard.set_init_success(true);
                guard.try_into_guarded()
            }
            _ => Err(crate::error::get_and_clear_last_mlx_error()
                .expect("MLX operation failed but no error was set")),
        }
    }
}

pub(crate) struct MaybeUninitArray {
    pub(crate) ptr: mlx_array,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitArray {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitArray {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_array_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitArray {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_array_free(self.ptr);
            }
        }
    }
}

impl Guard<Array> for MaybeUninitArray {
    type MutRawPtr = *mut mlx_array;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }
    
    fn try_into_guarded(self) -> Result<Array, Exception> {
        debug_assert!(self.init_success);
        Ok(Array {
            c_array: self.ptr,
        })
    }
}

impl Guarded for Array {
    type Guard = MaybeUninitArray;
}

pub(crate) struct MaybeUninitVectorArray {
    pub(crate) ptr: mlx_sys::mlx_vector_array,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitVectorArray {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitVectorArray {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_vector_array_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitVectorArray {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_vector_array_free(self.ptr);
            }
        }
    }
}

impl Guard<Vec<Array>> for MaybeUninitVectorArray {
    type MutRawPtr = *mut mlx_sys::mlx_vector_array;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }
    
    fn try_into_guarded(self) -> Result<Vec<Array>, Exception> {
        unsafe {
            let size = mlx_sys::mlx_vector_array_size(self.ptr);
            (0..size)
                .map(|i| {
                    Array::try_op(|res| {
                        mlx_sys::mlx_vector_array_get(res, self.ptr, i)
                    })
                })
                .collect()
        }
    }
}

impl Guard<(Array, Array)> for (MaybeUninitArray, MaybeUninitArray) {
    type MutRawPtr = (*mut mlx_array, *mut mlx_array);

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        (self.0.as_mut_raw_ptr(), self.1.as_mut_raw_ptr())
    }

    fn set_init_success(&mut self, success: bool) {
        self.0.set_init_success(success);
        self.1.set_init_success(success);
    }
    
    fn try_into_guarded(self) -> Result<(Array, Array), Exception> {
        Ok((self.0.try_into_guarded()?, self.1.try_into_guarded()?))
    }
}

impl Guarded for (Array, Array) {
    type Guard = (MaybeUninitArray, MaybeUninitArray);
}

impl Guard<(Array, Array, Array)> for (MaybeUninitArray, MaybeUninitArray, MaybeUninitArray) {
    type MutRawPtr = (*mut mlx_array, *mut mlx_array, *mut mlx_array);

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        (self.0.as_mut_raw_ptr(), self.1.as_mut_raw_ptr(), self.2.as_mut_raw_ptr())
    }

    fn set_init_success(&mut self, success: bool) {
        self.0.set_init_success(success);
        self.1.set_init_success(success);
        self.2.set_init_success(success);
    }
    
    fn try_into_guarded(self) -> Result<(Array, Array, Array), Exception> {
        Ok((self.0.try_into_guarded()?, self.1.try_into_guarded()?, self.2.try_into_guarded()?))
    }
}

impl Guarded for (Array, Array, Array) {
    type Guard = (MaybeUninitArray, MaybeUninitArray, MaybeUninitArray);
}

impl Guard<(Vec<Array>, Vec<Array>)> for (MaybeUninitVectorArray, MaybeUninitVectorArray) {
    type MutRawPtr = (*mut mlx_sys::mlx_vector_array, *mut mlx_sys::mlx_vector_array);

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        (self.0.as_mut_raw_ptr(), self.1.as_mut_raw_ptr())
    }

    fn set_init_success(&mut self, success: bool) {
        self.0.set_init_success(success);
        self.1.set_init_success(success);
    }
    
    fn try_into_guarded(self) -> Result<(Vec<Array>, Vec<Array>), Exception> {
        Ok((self.0.try_into_guarded()?, self.1.try_into_guarded()?))
    }
}

impl Guarded for (Vec<Array>, Vec<Array>) {
    type Guard = (MaybeUninitVectorArray, MaybeUninitVectorArray);
}

macro_rules! impl_guarded_for_primitive {
    ($type:ty) => {
        impl Guarded for $type {
            type Guard = $type;
        }

        impl Guard<$type> for $type {
            type MutRawPtr = *mut $type;

            fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
                self
            }

            fn set_init_success(&mut self, _: bool) { }

            fn try_into_guarded(self) -> Result<$type, Exception> {
                Ok(self)
            }
        }
    };

    ($($type:ty),*) => {
        $(impl_guarded_for_primitive!($type);)*
    };
}

impl_guarded_for_primitive!(
    bool,
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    f32,
    ()
);

impl Guarded for f16 {
    type Guard = float16_t;
}

impl Guard<f16> for float16_t {
    type MutRawPtr = *mut float16_t;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        self
    }

    fn set_init_success(&mut self, _: bool) { }

    fn try_into_guarded(self) -> Result<f16, Exception> {
        Ok(f16::from_bits(self.0))
    }
}

impl Guarded for bf16 {
    type Guard = bfloat16_t;
}

impl Guard<bf16> for bfloat16_t {
    type MutRawPtr = *mut bfloat16_t;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        self
    }

    fn set_init_success(&mut self, _: bool) { }

    fn try_into_guarded(self) -> Result<bf16, Exception> {
        Ok(bf16::from_bits(self))
    }
}

impl Guarded for complex64 {
    type Guard = __BindgenComplex<f32>;
}

impl Guard<complex64> for __BindgenComplex<f32> {
    type MutRawPtr = *mut __BindgenComplex<f32>;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        self
    }

    fn set_init_success(&mut self, _: bool) { }

    fn try_into_guarded(self) -> Result<complex64, Exception> {
        Ok(complex64::new(self.re, self.im))
    }
}