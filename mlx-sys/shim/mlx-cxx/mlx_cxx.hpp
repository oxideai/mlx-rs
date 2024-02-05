#pragma once

#include <memory>
#include <optional>

#include "mlx/stream.h"
#include "mlx/device.h"

namespace mlx_cxx {
    // Generic template constructor
    template <typename T, typename... Args> std::unique_ptr<T> new_unique(Args... args) {
        return std::unique_ptr<T>(new T(args...));
    }

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

    template <typename T>
    struct Optional {
        enum class Tag: uint8_t {
            None,
            Some
        };

        union Payload {
            std::nullopt_t none;
            T some;
        };

        Tag tag = Tag::None;
        Payload payload = Payload{ std::nullopt };
    };
}
