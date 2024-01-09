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

    bool array_item_bool(const array& arr, bool retain_graph);
    uint8_t array_item_uint8(const array& arr, bool retain_graph);
    uint16_t array_item_uint16(const array& arr, bool retain_graph);
    uint32_t array_item_uint32(const array& arr, bool retain_graph);
    uint64_t array_item_uint64(const array& arr, bool retain_graph);
    int8_t array_item_int8(const array& arr, bool retain_graph);
    int16_t array_item_int16(const array& arr, bool retain_graph);
    int32_t array_item_int32(const array& arr, bool retain_graph);
    int64_t array_item_int64(const array& arr, bool retain_graph);
    float16_t array_item_float16(const array& arr, bool retain_graph);
    bfloat16_t array_item_bfloat16(const array& arr, bool retain_graph);
    float array_item_float32(const array& arr, bool retain_graph);
    complex64_t array_item_complex64(const array& arr, bool retain_graph);
}
