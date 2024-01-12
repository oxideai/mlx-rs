#pragma once

#include <memory>

#include "mlx/stream.h"
#include "mlx/device.h"

namespace mlx_cxx {
    // Generic template constructor
    template <typename T, typename... Args> std::unique_ptr<T> new_unique(Args... args) {
        return std::unique_ptr<T>(new T(args...));
    }

    enum class StreamOrDeviceTag {
        Default,
        Stream,
        Device,
    };

    union StreamOrDevicePayload {
        std::monostate default_payload;
        mlx::core::Stream stream;
        mlx::core::Device device;
    };

    struct StreamOrDevice {
        StreamOrDeviceTag tag = StreamOrDeviceTag::Default;
        StreamOrDevicePayload payload = StreamOrDevicePayload{ std::monostate{} };

        std::variant<std::monostate, mlx::core::Stream, mlx::core::Device> to_variant();
    };
}
