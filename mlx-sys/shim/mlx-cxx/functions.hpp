#pragma once

#include "mlx/array.h"

namespace mlx_cxx {
    using CxxUnaryFn = std::function<mlx::core::array(const mlx::core::array &)>;
    using CxxMultiaryFn = std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array> &)>;

    /// @brief Multiple input and single output function. The function takes a vector of arrays and
    /// returns a single array
    using CxxMultiInputSingleOutputFn = std::function<mlx::core::array(const std::vector<mlx::core::array> &)>;

    /// @brief A function that takes two (pair) arrays and returns a single array
    using CxxPairInputSingleOutputFn = std::function<mlx::core::array(const mlx::core::array &, const mlx::core::array &)>;

    using CxxSingleInputPairOutputFn = std::function<std::pair<mlx::core::array, mlx::core::array>(const mlx::core::array&)>;

    using CxxVjpFn = std::function<std::vector<mlx::core::array>(
            const std::vector<mlx::core::array>&,
            const std::vector<mlx::core::array>&,
            const std::vector<mlx::core::array>&)>;
}