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
    std::unique_ptr<array> array_empty(mlx::core::Dtype dtype) {
        auto arr = mlx::core::array({}, dtype);
        return std::make_unique<array>(arr);
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

    void set_array_siblings(
        array& arr,
        std::unique_ptr<std::vector<array>> siblings,
        uint16_t position
    ) {
        arr.set_siblings(*siblings, position);
    }

    std::unique_ptr<std::vector<mlx::core::array>> array_outputs(mlx::core::array &arr) {
        return std::make_unique<std::vector<mlx::core::array>>(arr.outputs());
    }
}
