#pragma once

#include <string_view>

#include "mlx/dtype.h"
#include "mlx/utils.h"

#include "mlx-cxx/mlx_cxx.hpp"

#include "rust/cxx.h"

namespace mlx_cxx {
    // TODO: add binding to print format?

    struct StreamOrDevice {
        enum class Tag: uint8_t {
            Default,
            Stream,
            Device,
        };

        union Payload {
            std::monostate default_payload;
            mlx::core::Stream stream;
            mlx::core::Device device;
        };

        Tag tag = Tag::Default;
        Payload payload = Payload{ std::monostate{} };

        std::variant<std::monostate, mlx::core::Stream, mlx::core::Device> to_variant();
    };

    mlx::core::Dtype result_type(rust::Slice<const std::unique_ptr<mlx::core::array>> arrays);

    std::unique_ptr<std::vector<int>> broadcast_shapes(
        const std::vector<int>& s1,
        const std::vector<int>& s2);

    bool is_same_shape(rust::Slice<const std::unique_ptr<mlx::core::array>> arrays);

    // template<typename T>
    // void push_opaque(
    //     std::vector<T>& vec,
    //     std::unique_ptr<T> item);

    // template<typename T>
    // std::unique_ptr<T> pop_opaque(
    //     std::vector<T>& vec);

    // template<typename T>
    // std::unique_ptr<std::vector<T>> std_vec_from_slice(
    //     rust::Slice<const std::unique_ptr<T>> slice);
    
    template<typename T>
    void push_opaque(
        std::vector<T>& vec,
        std::unique_ptr<T> item) {
        vec.push_back(*item);
    }

    template<typename T>
    std::unique_ptr<std::vector<T>> std_vec_from_slice(
        rust::Slice<const std::unique_ptr<T>> slice) {
        std::vector<T> vec;
        for (auto& item : slice) {
            vec.push_back(*item);
        }
        return std::make_unique<std::vector<T>>(vec);
    }

    using OptionalArray = mlx_cxx::Optional<std::unique_ptr<mlx::core::array>>;

    std::optional<mlx::core::array> to_std_optional(const OptionalArray &opt);

    /* -------------------------------------------------------------------------- */
    /*                              string_view impl                              */
    /* -------------------------------------------------------------------------- */
    static_assert(sizeof(std::string_view) == 2 * sizeof(void *), "");
    static_assert(alignof(std::string_view) == alignof(void *), "");

    inline std::string_view string_view_from_str(rust::Str s) {
        return {s.data(), s.size()};
    }

    inline rust::Slice<const char> string_view_as_bytes(std::string_view s) {
        return {s.data(), s.size()};
    }
}