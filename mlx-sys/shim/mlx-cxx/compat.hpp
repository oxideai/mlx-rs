#pragma once

#include "mlx/array.h"
#include "mlx/transforms.h"

#include "mlx-cxx/functions.hpp"

#include "rust/cxx.h"

#include "mlx-sys/src/compat.rs.h"

namespace mlx_cxx
{
    // /* -------------------------------------------------------------------------- */
    // /*                     Bindings that accept rust funcionts                    */
    // /* -------------------------------------------------------------------------- */

    struct UnaryFn;
    struct MultiaryFn;
    struct VjpFn;
    struct MultiInputSingleOutputFn;
    struct PairInputSingleOutputFn;

    CxxUnaryFn make_unary_fn(const UnaryFn* f);

    CxxMultiaryFn make_multiary_fn(const MultiaryFn* f);

    CxxVjpFn make_vjp_fn(const VjpFn *f);

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> vjp(
        const MultiaryFn* fun,
        const std::vector<mlx::core::array> &primals,
        const std::vector<mlx::core::array> &cotangents);

    std::array<std::unique_ptr<mlx::core::array>, 2> vjp(
        const UnaryFn* fun,
        const mlx::core::array &primal,
        const mlx::core::array &cotangent);

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> jvp(
        const MultiaryFn* fun,
        const std::vector<mlx::core::array> &primals,
        const std::vector<mlx::core::array> &tangents);

    std::array<std::unique_ptr<mlx::core::array>, 2> jvp(
        const UnaryFn* fun,
        const mlx::core::array &primal,
        const mlx::core::array &tangent);

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const MultiaryFn* fun,
        const std::vector<int> &argnums);

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const MultiaryFn* fun,
        int argnum = 0);

    std::unique_ptr<CxxSingleInputPairOutputFn> value_and_grad(
        const UnaryFn* fun);

    std::unique_ptr<mlx::core::SimpleValueAndGradFn> value_and_grad(
        const MultiInputSingleOutputFn* fun,
        const std::vector<int> &argnums);

    std::unique_ptr<CxxMultiaryFn> grad(
        const MultiInputSingleOutputFn* fun,
        const std::vector<int> &argnums);

    std::unique_ptr<CxxMultiaryFn> grad(
        const MultiInputSingleOutputFn* fun,
        int argnum = 0);

    std::unique_ptr<CxxUnaryFn> grad(
        const UnaryFn* fun);

    std::unique_ptr<CxxUnaryFn> vmap(
        const UnaryFn* fun,
        int in_axis = 0,
        int out_axis = 0);

    std::unique_ptr<CxxPairInputSingleOutputFn> vmap(
        const PairInputSingleOutputFn* fun,
        int in_axis_a = 0,
        int in_axis_b = 0,
        int out_axis = 0);

    std::unique_ptr<CxxMultiaryFn> vmap(
        const MultiaryFn* fun,
        const std::vector<int> &in_axes = {},
        const std::vector<int> &out_axes = {});

    std::unique_ptr<CxxMultiaryFn> custom_vjp(
        const MultiaryFn* fun,
        const VjpFn* fun_vjp);

    std::unique_ptr<CxxMultiaryFn> checkpoint(
        const MultiaryFn* fun);

    // /* -------------------------------------------------------------------------- */

    std::unique_ptr<CxxMultiaryFn> compile(const MultiaryFn *fun);
}