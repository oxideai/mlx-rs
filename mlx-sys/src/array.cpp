#include "mlx-cxx/mlx_cxx.hpp"
#include "mlx-cxx/array.hpp"
#include "mlx-cxx/types.hpp"

#include "mlx-sys/src/types/float16.rs.h"
#include "mlx-sys/src/types/bfloat16.rs.h"
#include "mlx-sys/src/types/complex64.rs.h"

#include "mlx/types/half_types.h"
#include "mlx/types/complex.h"
#include "mlx/array.h"

namespace mlx_cxx {
    bool array_item_bool(array& arr, bool retain_graph) {
        return arr.item<bool>(retain_graph);
    }

    uint8_t array_item_uint8(array& arr, bool retain_graph) {
        return arr.item<uint8_t>(retain_graph);
    }

    uint16_t array_item_uint16(array& arr, bool retain_graph) {
        return arr.item<uint16_t>(retain_graph);
    }

    uint32_t array_item_uint32(array& arr, bool retain_graph) {
        return arr.item<uint32_t>(retain_graph);
    }

    uint64_t array_item_uint64(array& arr, bool retain_graph) {
        return arr.item<uint64_t>(retain_graph);
    }

    int8_t array_item_int8(array& arr, bool retain_graph) {
        return arr.item<int8_t>(retain_graph);
    }

    int16_t array_item_int16(array& arr, bool retain_graph) {
        return arr.item<int16_t>(retain_graph);
    }

    int32_t array_item_int32(array& arr, bool retain_graph) {
        return arr.item<int32_t>(retain_graph);
    }

    int64_t array_item_int64(array& arr, bool retain_graph) {
        return arr.item<int64_t>(retain_graph);
    }

    float16_t array_item_float16(array& arr, bool retain_graph) {
        return arr.item<mlx::core::float16_t>(retain_graph);
    }

    float array_item_float32(array& arr, bool retain_graph) {
        return arr.item<float>(retain_graph);
    }

    bfloat16_t array_item_bfloat16(array& arr, bool retain_graph) {
        return arr.item<mlx::core::bfloat16_t>(retain_graph);
    }

    complex64_t array_item_complex64(array& arr, bool retain_graph) {
        return arr.item<mlx::core::complex64_t>(retain_graph);
    }
}
