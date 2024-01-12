#include "mlx/utils.h"
#include "mlx-cxx/utils.hpp"

namespace mlx_cxx {
    std::unique_ptr<std::vector<int>> broadcast_shapes(
        const std::vector<int>& s1,
        const std::vector<int>& s2) {
        return std::make_unique<std::vector<int>>(mlx::core::broadcast_shapes(s1, s2));
    }
}