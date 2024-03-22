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

    std::unique_ptr<array> array_empty(mlx::core::Dtype dtype);
    std::unique_ptr<array> array_from_slice_bool(rust::Slice<const bool> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_uint8(rust::Slice<const uint8_t> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_uint16(rust::Slice<const uint16_t> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_uint32(rust::Slice<const uint32_t> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_uint64(rust::Slice<const uint64_t> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_int8(rust::Slice<const int8_t> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_int16(rust::Slice<const int16_t> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_int32(rust::Slice<const int32_t> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_int64(rust::Slice<const int64_t> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_float16(rust::Slice<const float16_t> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_bfloat16(rust::Slice<const bfloat16_t> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_float32(rust::Slice<const float> slice, std::unique_ptr<std::vector<int>> shape);
    std::unique_ptr<array> array_from_slice_complex64(rust::Slice<const complex64_t> slice, std::unique_ptr<std::vector<int>> shape);

    void set_array_siblings(
        array& arr,
        std::unique_ptr<std::vector<array>> siblings,
        uint16_t position
    );

    std::unique_ptr<std::vector<mlx::core::array>> array_outputs(mlx::core::array &arr);
}
