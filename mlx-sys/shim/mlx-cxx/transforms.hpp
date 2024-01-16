#pragma once

#include "mlx/array.h"

#include "rust/cxx.h"

// #include "mlx-sys/src/transforms.rs.h"

using MultiaryFn = std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array> &)>;

namespace mlx_cxx
{
    // typedef std::function<mlx::core::array(const mlx::core::array &)> UnaryFn;

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
        // const std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array> &)> &fun,
        const MultiaryFn &fun,
        rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
        rust::Slice<const std::unique_ptr<mlx::core::array>> cotangents);
}