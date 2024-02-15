#include "mlx/linalg.h"

#include "mlx-cxx/mlx_cxx.hpp"
#include "mlx-cxx/linalg.hpp"

namespace mlx_cxx {
    std::optional<std::vector<int>> to_std_optional(const OptionalAxis &opt) {
        switch (opt.tag) {
            case OptionalAxis::Tag::None:
                return std::nullopt;
            case OptionalAxis::Tag::Some:
                return *opt.payload.some;
        }
    }

    std::unique_ptr<mlx::core::array> norm_ord(
        const mlx::core::array &a,
        const double ord,
        const OptionalAxis &axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto axis_std = to_std_optional(axis);
        auto array = mlx::core::linalg::norm(a, ord, axis_std, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> norm_ord_axis(
        const mlx::core::array& a,
        const double ord,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::linalg::norm(a, ord, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> norm_str_ord(
        const mlx::core::array& a,
        const std::string& ord,
        const OptionalAxis &axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto axis_std = to_std_optional(axis);
        auto array = mlx::core::linalg::norm(a, ord, axis_std, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    // std::unique_ptr<mlx::core::array> norm_str_ord_axis(
    //     const mlx::core::array& a,
    //     const std::string& ord,
    //     int axis,
    //     bool keepdims,
    //     mlx_cxx::StreamOrDevice s)
    // {
    //     auto array = mlx::core::linalg::norm(a, ord, axis, keepdims, s.to_variant());
    //     return std::make_unique<mlx::core::array>(array);
    // }

    std::unique_ptr<mlx::core::array> norm(
        const mlx::core::array& a,
        const OptionalAxis &axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto axis_std = to_std_optional(axis);
        auto array = mlx::core::linalg::norm(a, axis_std, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> norm_axis(
        const mlx::core::array& a,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::linalg::norm(a, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::array<std::unique_ptr<mlx::core::array>, 2> qr(
        const mlx::core::array& a,
        mlx_cxx::StreamOrDevice s)
    {
        auto [q, r] = mlx::core::linalg::qr(a, s.to_variant());
        return {std::make_unique<mlx::core::array>(q), std::make_unique<mlx::core::array>(r)};
    }
}