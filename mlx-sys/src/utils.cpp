#include "mlx/utils.h"
#include "mlx/array.h"
#include "mlx-cxx/utils.hpp"

namespace mlx_cxx {
    std::variant<std::monostate, mlx::core::Stream, mlx::core::Device> StreamOrDevice::to_variant() {
        switch (tag) {
            case StreamOrDevice::Tag::Default:
                return payload.default_payload;
            case StreamOrDevice::Tag::Stream:
                return payload.stream;
            case StreamOrDevice::Tag::Device:
                return payload.device;
        }
    }

    mlx::core::Dtype result_type(rust::Slice<const std::unique_ptr<mlx::core::array>> arrays) {
        // Create a vector of arrays from the slice using the copy constructor
        std::vector<mlx::core::array> copy_constructed_arrays;
        for (auto& array : arrays) {
            copy_constructed_arrays.push_back(*array);
        }
        return mlx::core::result_type(copy_constructed_arrays);
    }

    std::unique_ptr<std::vector<int>> broadcast_shapes(
        const std::vector<int>& s1,
        const std::vector<int>& s2) {
        return std::make_unique<std::vector<int>>(mlx::core::broadcast_shapes(s1, s2));
    }

    bool is_same_shape(rust::Slice<const std::unique_ptr<mlx::core::array>> arrays) {
        // Create a vector of arrays from the slice using the copy constructor
        std::vector<mlx::core::array> copy_constructed_arrays;
        for (auto& array : arrays) {
            copy_constructed_arrays.push_back(*array);
        }
        return mlx::core::is_same_shape(copy_constructed_arrays);
    }

    // template<typename T>
    // void push_opaque(
    //     std::vector<T>& vec,
    //     std::unique_ptr<T> item) {
    //     vec.push_back(*item);
    // }

    // template<typename T>
    // std::unique_ptr<T> pop_opaque(
    //     std::vector<T>& vec) {
    //     auto item = vec.pop_back();
    //     return std::make_unique<T>(item);
    // }

    // template<typename T>
    // std::unique_ptr<std::vector<T>> std_vec_from_slice(
    //     rust::Slice<const std::unique_ptr<T>> slice) {
    //     std::vector<T> vec;
    //     for (auto& item : slice) {
    //         vec.push_back(*item);
    //     }
    //     return std::make_unique<std::vector<T>>(vec);
    // }

    std::optional<mlx::core::array> to_std_optional(const OptionalArray &opt)
    {
        switch (opt.tag)
        {
        case OptionalArray::Tag::None:
            return std::nullopt;
        case OptionalArray::Tag::Some:
            return *opt.payload.some;
        }
    }

}