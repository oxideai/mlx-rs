#include "mlx/transforms.h"

#include "mlx-cxx/transforms.hpp"

#include "mlx-sys/src/compat.rs.h"

#include "rust/cxx.h"

namespace mlx_cxx
{
    // TODO: should this use a mutable Slice?
    void eval(const std::vector<mlx::core::array> &outputs)
    {
        mlx::core::eval(outputs);
    }

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> vjp(
        const CxxMultiaryFn &fun,
        const std::vector<mlx::core::array> &primals,
        const std::vector<mlx::core::array> &cotangents)
    {
        auto result = mlx::core::vjp(fun, primals, cotangents);

        auto outputs = std::make_unique<std::vector<mlx::core::array>>(result.first);
        auto vjps = std::make_unique<std::vector<mlx::core::array>>(result.second);

        return {std::move(outputs), std::move(vjps)};
    }

    std::array<std::unique_ptr<mlx::core::array>, 2> vjp(
        const CxxUnaryFn &fun,
        const mlx::core::array &primal,
        const mlx::core::array &cotangent)
    {
        auto result = mlx::core::vjp(fun, primal, cotangent);
        auto first = std::make_unique<mlx::core::array>(result.first);
        auto second = std::make_unique<mlx::core::array>(result.second);

        return {std::move(first), std::move(second)};
    }

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> jvp(
        const CxxMultiaryFn &fun,
        const std::vector<mlx::core::array> &primals,
        const std::vector<mlx::core::array> &tangents)
    {
        auto result = mlx::core::jvp(fun, primals, tangents);

        auto outputs = std::make_unique<std::vector<mlx::core::array>>(result.first);
        auto vjps = std::make_unique<std::vector<mlx::core::array>>(result.second);

        return {std::move(outputs), std::move(vjps)};
    }

    std::array<std::unique_ptr<mlx::core::array>, 2> jvp(
        const CxxUnaryFn &fun,
        const mlx::core::array &primal,
        const mlx::core::array &tangent)
    {
        auto result = mlx::core::jvp(fun, primal, tangent);
        auto first = std::make_unique<mlx::core::array>(result.first);
        auto second = std::make_unique<mlx::core::array>(result.second);

        return {std::move(first), std::move(second)};
    }

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const CxxMultiaryFn &fun,
        const std::vector<int> &argnums)
    {
        return std::make_unique<mlx::core::ValueAndGradFn>(mlx::core::value_and_grad(fun, argnums));
    }

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const CxxMultiaryFn &fun,
        int argnum)
    {
        return std::make_unique<mlx::core::ValueAndGradFn>(mlx::core::value_and_grad(fun, argnum));
    }

    std::unique_ptr<CxxSingleInputPairOutputFn> value_and_grad(
        const CxxUnaryFn &fun)
    {
        return std::make_unique<CxxSingleInputPairOutputFn>(mlx::core::value_and_grad(fun));
    }

    std::unique_ptr<mlx::core::SimpleValueAndGradFn> value_and_grad(
        const CxxMultiInputSingleOutputFn &fun,
        const std::vector<int> &argnums)
    {
        return std::make_unique<mlx::core::SimpleValueAndGradFn>(mlx::core::value_and_grad(fun, argnums));
    }

    std::unique_ptr<CxxMultiaryFn> grad(
        const CxxMultiInputSingleOutputFn &fun,
        const std::vector<int> &argnums)
    {
        return std::make_unique<CxxMultiaryFn>(mlx::core::grad(fun, argnums));
    }

    std::unique_ptr<CxxMultiaryFn> grad(
        const CxxMultiInputSingleOutputFn &fun,
        int argnum)
    {
        return std::make_unique<CxxMultiaryFn>(mlx::core::grad(fun, argnum));
    }

    std::unique_ptr<CxxUnaryFn> grad(
        const CxxUnaryFn &fun)
    {
        return std::make_unique<CxxUnaryFn>(mlx::core::grad(fun));
    }

    std::unique_ptr<CxxUnaryFn> vmap(
        const CxxUnaryFn &fun,
        int in_axis,
        int out_axis)
    {
        return std::make_unique<CxxUnaryFn>(mlx::core::vmap(fun, in_axis, out_axis));
    }

    std::unique_ptr<CxxPairInputSingleOutputFn> vmap(
        const CxxPairInputSingleOutputFn &fun,
        int in_axis_a,
        int in_axis_b,
        int out_axis)
    {
        return std::make_unique<CxxPairInputSingleOutputFn>(mlx::core::vmap(fun, in_axis_a, in_axis_b, out_axis));
    }

    std::unique_ptr<CxxMultiaryFn> vmap(
        const CxxMultiaryFn &fun,
        const std::vector<int> &in_axes,
        const std::vector<int> &out_axes)
    {
        return std::make_unique<CxxMultiaryFn>(mlx::core::vmap(fun, in_axes, out_axes));
    }

    std::unique_ptr<CxxMultiaryFn> custom_vjp(
        std::unique_ptr<CxxMultiaryFn> fun,
        std::unique_ptr<CxxVjpFn> fun_vjp)
    {
        return std::make_unique<CxxMultiaryFn>(mlx::core::custom_vjp(*fun, *fun_vjp));
    }

    std::unique_ptr<CxxMultiaryFn> checkpoint(
        std::unique_ptr<CxxMultiaryFn> fun)
    {
        return std::make_unique<CxxMultiaryFn>(mlx::core::checkpoint(*fun));
    }
}