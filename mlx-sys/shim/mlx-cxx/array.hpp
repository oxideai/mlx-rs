#pragma once

#include "mlx/array.h"

using namespace mlx::core;

namespace mlx_cxx {
    void hello();

    // Naming convention: array_new_<dtype>(<value>)
    std::unique_ptr<array> array_new_bool(bool value);
}
