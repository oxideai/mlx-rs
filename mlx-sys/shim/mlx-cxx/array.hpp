#pragma once

#include <vector>

#include "mlx/array.h"

#include "mlx-cxx/types.hpp"

#include "mlx-sys/src/types/float16.rs.h"
#include "mlx-sys/src/types/bfloat16.rs.h"
#include "mlx-sys/src/types/complex64.rs.h"

#include "rust/cxx.h"

using namespace mlx::core;

namespace mlx_cxx
{
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

    bool array_item_bool(array &arr);
    uint8_t array_item_uint8(array &arr);
    uint16_t array_item_uint16(array &arr);
    uint32_t array_item_uint32(array &arr);
    uint64_t array_item_uint64(array &arr);
    int8_t array_item_int8(array &arr);
    int16_t array_item_int16(array &arr);
    int32_t array_item_int32(array &arr);
    int64_t array_item_int64(array &arr);
    float16_t array_item_float16(array &arr);
    bfloat16_t array_item_bfloat16(array &arr);
    float array_item_float32(array &arr);
    complex64_t array_item_complex64(array &arr);

    std::unique_ptr<array> array_from_slice_bool(rust::Slice<const bool> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_uint8(rust::Slice<const uint8_t> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_uint16(rust::Slice<const uint16_t> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_uint32(rust::Slice<const uint32_t> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_uint64(rust::Slice<const uint64_t> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_int8(rust::Slice<const int8_t> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_int16(rust::Slice<const int16_t> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_int32(rust::Slice<const int32_t> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_int64(rust::Slice<const int64_t> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_float16(rust::Slice<const float16_t> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_bfloat16(rust::Slice<const bfloat16_t> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_float32(rust::Slice<const float> slice, const std::vector<int> &shape);
    std::unique_ptr<array> array_from_slice_complex64(rust::Slice<const complex64_t> slice, const std::vector<int> &shape);
}
