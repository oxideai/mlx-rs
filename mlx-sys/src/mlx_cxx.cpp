#include "mlx-cxx/mlx_cxx.hpp"

namespace mlx_cxx {
    std::variant<std::monostate, mlx::core::Stream, mlx::core::Device> StreamOrDevice::to_variant() {
        switch (tag) {
            case StreamOrDeviceTag::Default:
                return payload.default_payload;
            case StreamOrDeviceTag::Stream:
                return payload.stream;
            case StreamOrDeviceTag::Device:
                return payload.device;
        }
    }
}