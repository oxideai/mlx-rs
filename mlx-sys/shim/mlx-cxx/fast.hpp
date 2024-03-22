#pragma once

#include "mlx/array.h"
#include "mlx/fast.h"

#include "mlx-cxx/mlx_cxx.hpp"
#include "mlx-cxx/utils.hpp"

namespace mlx_cxx::fast {
    using OptionalArray = mlx_cxx::Optional<std::unique_ptr<mlx::core::array>>;

    std::unique_ptr<mlx::core::array> rope(
        const mlx::core::array& x,
        int dims,
        bool traditional,
        float base,
        float scale,
        int offset,
        mlx_cxx::StreamOrDevice s);

    std::unique_ptr<mlx::core::array> scaled_dot_product_attention(
        const mlx::core::array & queries,
        const mlx::core::array & keys,
        const mlx::core::array & values,
        const float scale,
        const OptionalArray & mask,
        mlx_cxx::StreamOrDevice s
    );

    std::unique_ptr<mlx::core::array> rms_norm(
        const mlx::core::array & x,
        const mlx::core::array & weight,
        float eps,
        mlx_cxx::StreamOrDevice s
    );

    std::unique_ptr<mlx::core::array> layer_norm(
        const mlx::core::array & x,
        const OptionalArray & weight,
        const OptionalArray & bias,
        float eps,
        mlx_cxx::StreamOrDevice s
    );
}