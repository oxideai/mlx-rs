#pragma once

#include "mlx/array.h"

#include "mlx-cxx/types.hpp"
#include "mlx-sys/src/types/float16.rs.h"
#include "mlx-sys/src/types/bfloat16.rs.h"
#include "mlx-sys/src/types/complex64.rs.h"

using namespace mlx::core;

namespace mlx_cxx {
    // bool_,
    // uint8,
    // uint16,
    // uint32,
    // uint64,
    // int8,
    // int16,
    // int32,
    // int64,
    // float16,
    // float32,
    // bfloat16,
    // complex64,

    // // Naming convention: array_new_<dtype>(<value>)
    // std::unique_ptr<array> array_new_bool(bool value);

    std::unique_ptr<array> array_new_f16(mlx_cxx::f16 value);
    std::unique_ptr<array> array_new_bf16(mlx_cxx::bf16 value);
    std::unique_ptr<array> array_new_c64(mlx_cxx::c64 value);
}
