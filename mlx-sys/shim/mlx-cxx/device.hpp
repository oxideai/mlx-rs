#pragma once

#include <memory>

#include "mlx/device.h"

namespace mlx_cxx {
    using DeviceDeviceType = mlx::core::Device::DeviceType;

    mlx::core::Device new_device(DeviceDeviceType type, int index = 0) {
        return mlx::core::Device(type, index);
    }
}