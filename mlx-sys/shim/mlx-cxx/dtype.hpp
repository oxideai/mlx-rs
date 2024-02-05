#pragma once

#include "mlx/dtype.h"

namespace mlx_cxx {
    // We need type alias to support nested enum class
    using DtypeVal = mlx::core::Dtype::Val;
    using DtypeKind = mlx::core::Dtype::Kind;

    mlx::core::Dtype dtype_new(DtypeVal val, uint8_t size);

    mlx::core::Dtype dtype_bool_();
    
    mlx::core::Dtype dtype_uint8();
    mlx::core::Dtype dtype_uint16();
    mlx::core::Dtype dtype_uint32();
    mlx::core::Dtype dtype_uint64();

    mlx::core::Dtype dtype_int8();
    mlx::core::Dtype dtype_int16();
    mlx::core::Dtype dtype_int32();
    mlx::core::Dtype dtype_int64();

    mlx::core::Dtype dtype_float16();
    mlx::core::Dtype dtype_float32();
    mlx::core::Dtype dtype_bfloat16();
    mlx::core::Dtype dtype_complex64();

    std::unique_ptr<std::string> dtype_to_array_protocol(const mlx::core::Dtype& dtype);
}