#pragma once

#include "mlx/array.h"
#include "mlx/transforms.h"

#include "rust/cxx.h"

#include "mlx-sys/src/function.rs.h"

namespace mlx_cxx
{
    using UnaryCxxFn = std::function<mlx::core::array(const mlx::core::array &)>;
    using MultiaryCxxFn = std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array> &)>;

    /// @brief Multiple input and single output function. The function takes a vector of arrays and
    /// returns a single array
    using MisoCxxFn = std::function<mlx::core::array(const std::vector<mlx::core::array> &)>;

    /// @brief A function that takes two (pair) arrays and returns a single array
    using PisoCxxFn = std::function<mlx::core::array(const mlx::core::array &, const mlx::core::array &)>;

    using SipoCxxFn = std::function<std::pair<mlx::core::array, mlx::core::array>(const mlx::core::array&)>;

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

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const MultiaryCxxFn &fun,
        const std::vector<int> &argnums);

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const MultiaryCxxFn &fun, 
        int argnum = 0);

    std::unique_ptr<SipoCxxFn> valud_and_grad(
        const UnaryCxxFn &fun);

    std::unique_ptr<mlx::core::SimpleValueAndGradFn> value_and_grad(
        const MisoCxxFn &fun,
        const std::vector<int> &argnums);

    std::unique_ptr<MultiaryCxxFn> grad(
        const MisoCxxFn &fun,
        const std::vector<int> &argnums);

    std::unique_ptr<MultiaryCxxFn> grad(
        const MisoCxxFn &fun,
        int argnum = 0);

    std::unique_ptr<UnaryCxxFn> grad(
        const UnaryCxxFn &fun);

    std::unique_ptr<UnaryCxxFn> vmap(
        const UnaryCxxFn &fun,
        int in_axis = 0,
        int out_axis = 0);

    std::unique_ptr<PisoCxxFn> vmap(
        const PisoCxxFn &fun,
        int in_axis_a = 0,
        int in_axis_b = 0,
        int out_axis = 0);

    std::unique_ptr<MultiaryCxxFn> vmap(
        const MultiaryCxxFn &fun,
        const std::vector<int> &in_axes = {},
        const std::vector<int> &out_axes = {});

    /* -------------------------------------------------------------------------- */
    /*                     Bindings that accept rust funcionts                    */
    /* -------------------------------------------------------------------------- */

    // TODO: This is for test only. Remove later
    int accept_rust_unary_fn(const mlx_cxx::UnaryFn &f);

    UnaryCxxFn make_unary_fn(UnaryFn* f);

    MultiaryCxxFn make_multiary_fn(MultiaryFn* f);

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> vjp(
        const MultiaryFn* fun,
        rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
        rust::Slice<const std::unique_ptr<mlx::core::array>> cotangents);

    std::array<std::unique_ptr<mlx::core::array>, 2> vjp(
        const UnaryFn* fun,
        const mlx::core::array &primal,
        const mlx::core::array &cotangent);

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> jvp(
        const MultiaryFn* fun,
        rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
        rust::Slice<const std::unique_ptr<mlx::core::array>> tangents);

    std::array<std::unique_ptr<mlx::core::array>, 2> jvp(
        const UnaryFn* fun,
        const mlx::core::array &primal,
        const mlx::core::array &tangent);

}