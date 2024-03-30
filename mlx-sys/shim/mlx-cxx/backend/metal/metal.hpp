#pragma once

#include "mlx/backend/metal/metal.h"

namespace mlx_cxx::metal {
    bool start_capture(std::unique_ptr<std::string> path);
}