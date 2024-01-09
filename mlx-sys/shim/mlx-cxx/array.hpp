#pragma once

#include <vector>

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

    // // Naming convention: array_<method>_<dtype>(<value>)
    // std::unique_ptr<array> array_new_bool(bool value);

    std::unique_ptr<array> array_new_f16(mlx_cxx::float16_t value);
    std::unique_ptr<array> array_new_bf16(mlx_cxx::bfloat16_t value);
    std::unique_ptr<array> array_new_c64(mlx_cxx::complex64_t value);

    bool array_item_bool(const array& arr, bool retain_graph);
    uint8_t array_item_uint8(const array& arr, bool retain_graph);
    uint16_t array_item_uint16(const array& arr, bool retain_graph);
    uint32_t array_item_uint32(const array& arr, bool retain_graph);
    uint64_t array_item_uint64(const array& arr, bool retain_graph);
    int8_t array_item_int8(const array& arr, bool retain_graph);
    int16_t array_item_int16(const array& arr, bool retain_graph);
    int32_t array_item_int32(const array& arr, bool retain_graph);
    int64_t array_item_int64(const array& arr, bool retain_graph);
    mlx_cxx::float16_t array_item_float64(const array& arr, bool retain_graph);
}
