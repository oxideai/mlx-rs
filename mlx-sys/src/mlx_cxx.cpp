#include "mlx-cxx/mlx_cxx.hpp"

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
}