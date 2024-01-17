#pragma once

#include "mlx/array.h"

#include "rust/cxx.h"

#include "mlx-sys/src/function.rs.h"

namespace mlx_cxx
{
    using UnaryCxxFn = std::function<mlx::core::array(const mlx::core::array &)>;
    using MultiaryCxxFn = std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array> &)>;

    // using ValueAndGradFn = std::function<std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
    //     const std::vector<mlx::core::array> &)>;
    // using SimpleValueAndGradFn = std::function<std::pair<mlx::core::array, std::vector<mlx::core::array>>(
    //     const std::vector<mlx::core::array> &)>;

    // int execute_callback(const mlx_cxx::DynFn &f, int args);

    void simplify(rust::Slice<const std::unique_ptr<mlx::core::array>> outputs);

    // TODO: what about the templated simplify?

    void eval(rust::Slice<const std::unique_ptr<mlx::core::array>> outputs);

    // TODO: what about the templated eval?

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> vjp(
        const MultiaryCxxFn &fun,
        rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
        rust::Slice<const std::unique_ptr<mlx::core::array>> cotangents);

    std::array<std::unique_ptr<mlx::core::array>, 2> vjp(
        const UnaryCxxFn &fun,
        const mlx::core::array &primal,
        const mlx::core::array &cotangent);

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> jvp(
        const MultiaryCxxFn &fun,
        rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
        rust::Slice<const std::unique_ptr<mlx::core::array>> tangents);

    std::array<std::unique_ptr<mlx::core::array>, 2> jvp(
        const UnaryCxxFn &fun,
        const mlx::core::array &primal,
        const mlx::core::array &tangent);

    // TODO: This is for test only. Remove later
    int accept_rust_unary_fn(const mlx_cxx::UnaryFn &f);

    UnaryCxxFn make_unary_fn(UnaryFn* f);

    // std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> vjp(
    //     const MultiaryFn &fun,
    //     rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
    //     rust::Slice<const std::unique_ptr<mlx::core::array>> cotangents);

    std::array<std::unique_ptr<mlx::core::array>, 2> vjp(
        const UnaryFn* fun,
        const mlx::core::array &primal,
        const mlx::core::array &cotangent);

}