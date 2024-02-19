use cxx::UniquePtr;
use mlx_sys::{array::ffi, types::{float16::float16_t, bfloat16::bfloat16_t, complex64::complex64_t}};

pub struct Array {
    inner: UniquePtr<ffi::array>,
}

impl From<bool> for Array {
    fn from(value: bool) -> Self {
        Self {
            inner: ffi::array_new_bool(value),
        }
    }
}

impl From<i8> for Array {
    fn from(value: i8) -> Self {
        Self {
            inner: ffi::array_new_int8(value),
        }
    }
}

impl From<i16> for Array {
    fn from(value: i16) -> Self {
        Self {
            inner: ffi::array_new_int16(value),
        }
    }
}

impl From<i32> for Array {
    fn from(value: i32) -> Self {
        Self {
            inner: ffi::array_new_int32(value),
        }
    }
}

impl From<i64> for Array {
    fn from(value: i64) -> Self {
        Self {
            inner: ffi::array_new_int64(value),
        }
    }
}

impl From<u8> for Array {
    fn from(value: u8) -> Self {
        Self {
            inner: ffi::array_new_uint8(value),
        }
    }
}

impl From<u16> for Array {
    fn from(value: u16) -> Self {
        Self {
            inner: ffi::array_new_uint16(value),
        }
    }
}

impl From<u32> for Array {
    fn from(value: u32) -> Self {
        Self {
            inner: ffi::array_new_uint32(value),
        }
    }
}

impl From<u64> for Array {
    fn from(value: u64) -> Self {
        Self {
            inner: ffi::array_new_uint64(value),
        }
    }
}

impl From<f32> for Array {
    fn from(value: f32) -> Self {
        Self {
            inner: ffi::array_new_float32(value),
        }
    }
}

impl From<float16_t> for Array {
    fn from(value: float16_t) -> Self {
        Self {
            inner: ffi::array_new_float16(value),
        }
    }
}

#[cfg(feature = "half")]
impl From<half::f16> for Array {
    fn from(value: half::f16) -> Self {
        let val = float16_t {
            bits: value.to_bits(),
        };
        Self {
            inner: ffi::array_new_float16(val),
        }
    }
}

impl From<bfloat16_t> for Array {
    fn from(value: bfloat16_t) -> Self {
        Self {
            inner: ffi::array_new_bfloat16(value),
        }
    }
}

#[cfg(feature = "half")]
impl From<bfloat16_t> for Array {
    fn from(value: bfloat16_t) -> Self {
        let val = half::f16::from_bits(value.bits);
        Self {
            inner: ffi::array_new_float16(val.into()),
        }
    }
}

