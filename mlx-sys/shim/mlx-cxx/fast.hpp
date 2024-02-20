#pragma once

#include "mlx/array.h"
#include "mlx/fast.h"

#include "mlx-cxx/mlx_cxx.hpp"

namespace mlx_cxx::fast {
    std::unique_ptr<mlx::core::array> rope(
        const mlx::core::array& x,
        int dims,
        bool traditional,
        float base,
        float scale,
        int offset,
        mlx_cxx::StreamOrDevice s);
}