#pragma once

#include "mlx/dtype.h"

namespace mlx_cxx {
    // We need type alias to support nested enum class
    using DtypeVal = mlx::core::Dtype::Val;
    using DtypeKind = mlx::core::Dtype::Kind;

    mlx::core::Dtype dtype_new(DtypeVal val, uint8_t size);

    constexpr mlx::core::Dtype dtype_bool_();
    
    constexpr mlx::core::Dtype dtype_uint8();
    constexpr mlx::core::Dtype dtype_uint16();
    constexpr mlx::core::Dtype dtype_uint32();
    constexpr mlx::core::Dtype dtype_uint64();

    constexpr mlx::core::Dtype dtype_int8();
    constexpr mlx::core::Dtype dtype_int16();
    constexpr mlx::core::Dtype dtype_int32();
    constexpr mlx::core::Dtype dtype_int64();

    constexpr mlx::core::Dtype dtype_float16();
    constexpr mlx::core::Dtype dtype_float32();
    constexpr mlx::core::Dtype dtype_bfloat16();
    constexpr mlx::core::Dtype dtype_complex64();

    std::unique_ptr<std::string> dtype_to_array_protocol(const mlx::core::Dtype& dtype);
}