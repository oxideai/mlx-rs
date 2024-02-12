#pragma once

#include "mlx/array.h"
#include "mlx/transforms.h"

#include "mlx-cxx/functions.hpp"

#include "rust/cxx.h"

#include "mlx-sys/src/compat.rs.h"

namespace mlx_cxx
{
    void eval(rust::Slice<const std::unique_ptr<mlx::core::array>> outputs);

    // TODO: what about the templated eval?

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> vjp(
        const CxxMultiaryFn &fun,
        rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
        rust::Slice<const std::unique_ptr<mlx::core::array>> cotangents);

    std::array<std::unique_ptr<mlx::core::array>, 2> vjp(
        const CxxUnaryFn &fun,
        const mlx::core::array &primal,
        const mlx::core::array &cotangent);

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> jvp(
        const CxxMultiaryFn &fun,
        rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
        rust::Slice<const std::unique_ptr<mlx::core::array>> tangents);

    std::array<std::unique_ptr<mlx::core::array>, 2> jvp(
        const CxxUnaryFn &fun,
        const mlx::core::array &primal,
        const mlx::core::array &tangent);

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const CxxMultiaryFn &fun,
        const std::vector<int> &argnums);

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const CxxMultiaryFn &fun, 
        int argnum = 0);

    std::unique_ptr<CxxSingleInputPairOutputFn> value_and_grad(
        const CxxUnaryFn &fun);

    std::unique_ptr<mlx::core::SimpleValueAndGradFn> value_and_grad(
        const CxxMultiInputSingleOutputFn &fun,
        const std::vector<int> &argnums);

    std::unique_ptr<CxxMultiaryFn> grad(
        const CxxMultiInputSingleOutputFn &fun,
        const std::vector<int> &argnums);

    std::unique_ptr<CxxMultiaryFn> grad(
        const CxxMultiInputSingleOutputFn &fun,
        int argnum = 0);

    std::unique_ptr<CxxUnaryFn> grad(
        const CxxUnaryFn &fun);

    std::unique_ptr<CxxUnaryFn> vmap(
        const CxxUnaryFn &fun,
        int in_axis = 0,
        int out_axis = 0);

    std::unique_ptr<CxxPairInputSingleOutputFn> vmap(
        const CxxPairInputSingleOutputFn &fun,
        int in_axis_a = 0,
        int in_axis_b = 0,
        int out_axis = 0);

    std::unique_ptr<CxxMultiaryFn> vmap(
        const CxxMultiaryFn &fun,
        const std::vector<int> &in_axes = {},
        const std::vector<int> &out_axes = {});

    /**
     * Return the results of calling fun with args but if their vjp is computed it
     * will be computed by fun_vjp.
     */
    std::unique_ptr<CxxMultiaryFn> custom_vjp(
        std::unique_ptr<CxxMultiaryFn> fun,
        std::unique_ptr<CxxVjpFn> fun_vjp);

    /**
     * Checkpoint the gradient of a function. Namely, discard all intermediate
     * state and recalculate it when we need to compute the gradient.
     */
    std::unique_ptr<CxxMultiaryFn> checkpoint(
        std::unique_ptr<CxxMultiaryFn> fun);


    // /* -------------------------------------------------------------------------- */
    // /*                     Bindings that accept rust funcionts                    */
    // /* -------------------------------------------------------------------------- */

    // CxxUnaryFn make_unary_fn(const UnaryFn* f);

    // CxxMultiaryFn make_multiary_fn(const MultiaryFn* f);

    // CxxVjpFn make_vjp_fn(const VjpFn *f);

    // std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> vjp(
    //     const MultiaryFn* fun,
    //     rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
    //     rust::Slice<const std::unique_ptr<mlx::core::array>> cotangents);

    // std::array<std::unique_ptr<mlx::core::array>, 2> vjp(
    //     const UnaryFn* fun,
    //     const mlx::core::array &primal,
    //     const mlx::core::array &cotangent);

    // std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> jvp(
    //     const MultiaryFn* fun,
    //     rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
    //     rust::Slice<const std::unique_ptr<mlx::core::array>> tangents);

    // std::array<std::unique_ptr<mlx::core::array>, 2> jvp(
    //     const UnaryFn* fun,
    //     const mlx::core::array &primal,
    //     const mlx::core::array &tangent);

    // std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
    //     const MultiaryFn* fun,
    //     const std::vector<int> &argnums);

    // std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
    //     const MultiaryFn* fun,
    //     int argnum = 0);

    // std::unique_ptr<CxxSingleInputPairOutputFn> value_and_grad(
    //     const UnaryFn* fun);

    // std::unique_ptr<mlx::core::SimpleValueAndGradFn> value_and_grad(
    //     const MultiInputSingleOutputFn* fun,
    //     const std::vector<int> &argnums);

    // std::unique_ptr<CxxMultiaryFn> grad(
    //     const MultiInputSingleOutputFn* fun,
    //     const std::vector<int> &argnums);

    // std::unique_ptr<CxxMultiaryFn> grad(
    //     const MultiInputSingleOutputFn* fun,
    //     int argnum = 0);

    // std::unique_ptr<CxxUnaryFn> grad(
    //     const UnaryFn* fun);

    // std::unique_ptr<CxxUnaryFn> vmap(
    //     const UnaryFn* fun,
    //     int in_axis = 0,
    //     int out_axis = 0);

    // std::unique_ptr<CxxPairInputSingleOutputFn> vmap(
    //     const PairInputSingleOutputFn* fun,
    //     int in_axis_a = 0,
    //     int in_axis_b = 0,
    //     int out_axis = 0);

    // std::unique_ptr<CxxMultiaryFn> vmap(
    //     const MultiaryFn* fun,
    //     const std::vector<int> &in_axes = {},
    //     const std::vector<int> &out_axes = {});

    // std::unique_ptr<CxxMultiaryFn> custom_vjp(
    //     const MultiaryFn* fun,
    //     const VjpFn* fun_vjp);

    // std::unique_ptr<CxxMultiaryFn> checkpoint(
    //     const MultiaryFn* fun);

    // /* -------------------------------------------------------------------------- */

    // std::unique_ptr<CxxMultiaryFn> compile(const MultiaryFn *fun);
}