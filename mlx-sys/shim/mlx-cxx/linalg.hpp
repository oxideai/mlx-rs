#pragma once

#include "mlx/linalg.h"

#include "mlx-cxx/mlx_cxx.hpp"
#include "mlx-cxx/utils.hpp"

namespace mlx_cxx
{
    using OptionalAxis = mlx_cxx::Optional<std::unique_ptr<std::vector<int>>>;

    std::optional<std::vector<int>> to_std_optional(const OptionalAxis &opt);

    std::unique_ptr<mlx::core::array> norm_ord(
        const mlx::core::array &a,
        const double ord,
        const OptionalAxis &axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> norm_ord_axis(
        const mlx::core::array &a,
        const double ord,
        int axis,
        bool keepdims = false,
        StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> norm_str_ord(
        const mlx::core::array &a,
        std::string_view ord,
        const OptionalAxis &axis,
        bool keepdims = false,
        StreamOrDevice s = {});

    // std::unique_ptr<mlx::core::array> norm_str_ord_axis(
    //     const mlx::core::array& a,
    //     const std::string& ord,
    //     int axis,
    //     bool keepdims = false,
    //     StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> norm(
        const mlx::core::array &a,
        const OptionalAxis &axis,
        bool keepdims = false,
        StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> norm_axis(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        StreamOrDevice s = {});

    std::array<std::unique_ptr<mlx::core::array>, 2> qr(
        const mlx::core::array &a,
        StreamOrDevice s = {});

    std::unique_ptr<std::vector<mlx::core::array>> svd(
        const mlx::core::array &a,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> inv(const mlx::core::array &a, mlx_cxx::StreamOrDevice s);
}