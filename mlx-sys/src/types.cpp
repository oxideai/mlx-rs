#include "mlx-cxx/types.hpp"

namespace mlx_cxx {
    mlx::core::complex64_t complex64(float v, float u) {
        return mlx::core::complex64_t(v, u);
    }
}