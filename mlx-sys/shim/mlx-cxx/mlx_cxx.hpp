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
