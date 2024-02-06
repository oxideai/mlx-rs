#include "mlx/transforms.h"

#include "mlx-cxx/transforms.hpp"

#include "mlx-sys/src/transforms/compat.rs.h"

#include "rust/cxx.h"

namespace mlx_cxx
{
    std::unique_ptr<CxxMultiaryFn> compile(const CxxMultiaryFn &fun)
    {
        return std::make_unique<CxxMultiaryFn>(mlx::core::compile(fun));
    }

    // TODO: should this use a mutable Slice?
    void eval(rust::Slice<const std::unique_ptr<mlx::core::array>> outputs)
    {
        auto outputs_vec = std::vector<mlx::core::array>{};
        for (auto &output : outputs)
        {
            outputs_vec.push_back(*output);
        }

        mlx::core::eval(outputs_vec);
    }

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> vjp(
        const CxxMultiaryFn &fun,
        rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
        rust::Slice<const std::unique_ptr<mlx::core::array>> cotangents)
    {
        auto primals_vec = std::vector<mlx::core::array>{};
        for (auto &primal : primals)
        {
            primals_vec.push_back(*primal);
        }

        auto cotangents_vec = std::vector<mlx::core::array>{};
        for (auto &cotangent : cotangents)
        {
            cotangents_vec.push_back(*cotangent);
        }

        auto result = mlx::core::vjp(fun, primals_vec, cotangents_vec);

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
        rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
        rust::Slice<const std::unique_ptr<mlx::core::array>> tangents)
    {
        auto primals_vec = std::vector<mlx::core::array>{};
        for (auto &primal : primals)
        {
            primals_vec.push_back(*primal);
        }

        auto tangents_vec = std::vector<mlx::core::array>{};
        for (auto &tangent : tangents)
        {
            tangents_vec.push_back(*tangent);
        }

        auto result = mlx::core::jvp(fun, primals_vec, tangents_vec);

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

    /* -------------------------------------------------------------------------- */
    /*                     Bindings that accept rust funcionts                    */
    /* -------------------------------------------------------------------------- */

    CxxUnaryFn make_unary_fn(const UnaryFn *f)
    {
        return [fun = std::move(f)](const mlx::core::array &arg)
        {
            auto ptr = mlx_cxx::execute_unary_fn(*fun, arg);
            return *ptr;
        };
    }

    CxxMultiaryFn make_multiary_fn(const MultiaryFn *f)
    {
        return [fun = std::move(f)](const std::vector<mlx::core::array> &args)
        {
            auto ptr = mlx_cxx::execute_multiary_fn(*fun, args);
            return *ptr;
        };
    }

    CxxMultiInputSingleOutputFn make_multi_input_single_output_fn(const MultiInputSingleOutputFn *f)
    {
        return [fun = std::move(f)](const std::vector<mlx::core::array> &args)
        {
            auto ptr = mlx_cxx::execute_multi_input_single_output_fn(*fun, args);
            return *ptr;
        };
    }

    CxxPairInputSingleOutputFn make_pair_input_single_output_fn(const PairInputSingleOutputFn *f)
    {
        return [fun = std::move(f)](const mlx::core::array &a, const mlx::core::array &b)
        {
            auto ptr = mlx_cxx::execute_pair_input_single_output_fn(*fun, a, b);
            return *ptr;
        };
    }

    CxxVjpFn make_vjp_fn(const VjpFn *f)
    {
        return [fun = std::move(f)](const std::vector<mlx::core::array>& arg1, const std::vector<mlx::core::array>& arg2, const std::vector<mlx::core::array>& arg3)
        {
            auto ptr = mlx_cxx::execute_vjp_fn(*fun, arg1, arg2, arg3);
            return *ptr;
        };
    }

    std::unique_ptr<CxxMultiaryFn> compile(const MultiaryFn *fun)
    {
        return mlx_cxx::compile(make_multiary_fn(fun));
    }

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> vjp(
    const MultiaryFn* fun,
    rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
    rust::Slice<const std::unique_ptr<mlx::core::array>> cotangents)
    {
        return mlx_cxx::vjp(make_multiary_fn(fun), primals, cotangents);
    }

    std::array<std::unique_ptr<mlx::core::array>, 2> vjp(
        const UnaryFn *fun,
        const mlx::core::array &primal,
        const mlx::core::array &cotangent)
    {
        return mlx_cxx::vjp(make_unary_fn(fun), primal, cotangent);
    }

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> jvp(
        const MultiaryFn* fun,
        rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
        rust::Slice<const std::unique_ptr<mlx::core::array>> tangents)
    {
        return mlx_cxx::jvp(make_multiary_fn(fun), primals, tangents);
    }

    std::array<std::unique_ptr<mlx::core::array>, 2> jvp(
        const UnaryFn* fun,
        const mlx::core::array &primal,
        const mlx::core::array &tangent)
    {
        return mlx_cxx::jvp(make_unary_fn(fun), primal, tangent);
    }

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const MultiaryFn* fun,
        const std::vector<int> &argnums)
    {
        return mlx_cxx::value_and_grad(make_multiary_fn(fun), argnums);
    }

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const MultiaryFn* fun,
        int argnum)
    {
        return mlx_cxx::value_and_grad(make_multiary_fn(fun), argnum);
    }

    std::unique_ptr<CxxSingleInputPairOutputFn> value_and_grad(
        const UnaryFn* fun)
    {
        return mlx_cxx::value_and_grad(make_unary_fn(fun));
    }

    std::unique_ptr<mlx::core::SimpleValueAndGradFn> value_and_grad(
        const MultiInputSingleOutputFn* fun,
        const std::vector<int> &argnums)
    {
        return mlx_cxx::value_and_grad(make_multi_input_single_output_fn(fun), argnums);
    }

    std::unique_ptr<CxxMultiaryFn> grad(
        const MultiInputSingleOutputFn* fun,
        const std::vector<int> &argnums)
    {
        return mlx_cxx::grad(make_multi_input_single_output_fn(fun), argnums);
    }

    std::unique_ptr<CxxMultiaryFn> grad(
        const MultiInputSingleOutputFn* fun,
        int argnum)
    {
        return mlx_cxx::grad(make_multi_input_single_output_fn(fun), argnum);
    }

    std::unique_ptr<CxxUnaryFn> grad(
        const UnaryFn* fun)
    {
        return mlx_cxx::grad(make_unary_fn(fun));
    }

    std::unique_ptr<CxxUnaryFn> vmap(
        const UnaryFn* fun,
        int in_axis,
        int out_axis)
    {
        return mlx_cxx::vmap(make_unary_fn(fun), in_axis, out_axis);
    }

    std::unique_ptr<CxxPairInputSingleOutputFn> vmap(
        const PairInputSingleOutputFn* fun,
        int in_axis_a,
        int in_axis_b,
        int out_axis)
    {
        return mlx_cxx::vmap(make_pair_input_single_output_fn(fun), in_axis_a, in_axis_b, out_axis);
    }

    std::unique_ptr<CxxMultiaryFn> vmap(
        const MultiaryFn* fun,
        const std::vector<int> &in_axes,
        const std::vector<int> &out_axes)
    {
        return mlx_cxx::vmap(make_multiary_fn(fun), in_axes, out_axes);
    }

    std::unique_ptr<CxxMultiaryFn> custom_vjp(
        const MultiaryFn* fun,
        const VjpFn* fun_vjp)
    {
        auto cxx_fun = make_multiary_fn(fun);
        auto cxx_vjp_fun = make_vjp_fn(fun_vjp);
        return std::make_unique<CxxMultiaryFn>(mlx::core::custom_vjp(cxx_fun, cxx_vjp_fun));
    }

    /* -------------------------------------------------------------------------- */
}