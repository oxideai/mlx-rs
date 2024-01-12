#pragma once

#include "mlx/utils.h"

namespace mlx_cxx {
    std::unique_ptr<std::vector<int>> broadcast_shapes(
        const std::vector<int>& s1,
        const std::vector<int>& s2);
}