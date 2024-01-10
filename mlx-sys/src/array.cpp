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

    std::unique_ptr<array> array_from_slice_bool(
        rust::Slice<const bool> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::bool_);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_uint8(
        rust::Slice<const uint8_t> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::uint8);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_uint16(
        rust::Slice<const uint16_t> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::uint16);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_uint32(
        rust::Slice<const uint32_t> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::uint32);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_uint64(
        rust::Slice<const uint64_t> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::uint64);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_int8(
        rust::Slice<const int8_t> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::int8);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_int16(
        rust::Slice<const int16_t> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::int16);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_int32(
        rust::Slice<const int32_t> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::int32);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_int64(
        rust::Slice<const int64_t> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::int64);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_float16(
        rust::Slice<const float16_t> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::float16);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_bfloat16(
        rust::Slice<const bfloat16_t> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::bfloat16);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_float32(
        rust::Slice<const float> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::float32);
        return std::make_unique<array>(arr);
    }

    std::unique_ptr<array> array_from_slice_complex64(
        rust::Slice<const complex64_t> slice,
        const std::vector<int>& shape
    ) {
        array arr = array(slice.begin(), shape, mlx::core::complex64);
        return std::make_unique<array>(arr);
    }
}
