#include "mlx-cxx/dtype.hpp"

namespace mlx_cxx {
    mlx::core::Dtype dtype_new(DtypeVal val, uint8_t size) {
        return mlx::core::Dtype(val, size);
    }

    mlx::core::Dtype dtype_bool_() {
        return mlx::core::bool_;
    }

    mlx::core::Dtype dtype_uint8() {
        return mlx::core::uint8;
    }

    mlx::core::Dtype dtype_uint16() {
        return mlx::core::uint16;
    }

    mlx::core::Dtype dtype_uint32() {
        return mlx::core::uint32;
    }

    mlx::core::Dtype dtype_uint64() {
        return mlx::core::uint64;
    }

    mlx::core::Dtype dtype_int8() {
        return mlx::core::int8;
    }

    mlx::core::Dtype dtype_int16() {
        return mlx::core::int16;
    }

    mlx::core::Dtype dtype_int32() {
        return mlx::core::int32;
    }

    mlx::core::Dtype dtype_int64() {
        return mlx::core::int64;
    }

    mlx::core::Dtype dtype_float16() {
        return mlx::core::float16;
    }

    mlx::core::Dtype dtype_float32() {
        return mlx::core::float32;
    }

    mlx::core::Dtype dtype_bfloat16() {
        return mlx::core::bfloat16;
    }

    mlx::core::Dtype dtype_complex64() {
        return mlx::core::complex64;
    }

    std::unique_ptr<std::string> dtype_to_array_protocol(const mlx::core::Dtype& dtype) {
        return std::make_unique<std::string>(mlx::core::dtype_to_array_protocol(dtype));
    }
}