#pragma once

#include "mlx-cxx/backend/metal/metal.hpp"

namespace mlx_cxx::metal {
    bool start_capture(std::unique_ptr<std::string> path) {
        return mlx::core::metal::start_capture(*path);
    }
}