#include "mlx-cxx/transforms.hpp"
#include "mlx-cxx/compile.hpp"
#include "mlx-cxx/compat.hpp"

namespace mlx_cxx
{
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
        return [fun = std::move(f)](const std::vector<mlx::core::array> &arg1, const std::vector<mlx::core::array> &arg2, const std::vector<mlx::core::array> &arg3)
        {
            auto ptr = mlx_cxx::execute_vjp_fn(*fun, arg1, arg2, arg3);
            return *ptr;
        };
    }

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> vjp(
        const MultiaryFn *fun,
        const std::vector<mlx::core::array> &primals,
        const std::vector<mlx::core::array> &cotangents)
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
        const MultiaryFn *fun,
        const std::vector<mlx::core::array> &primals,
        const std::vector<mlx::core::array> &tangents)
    {
        return mlx_cxx::jvp(make_multiary_fn(fun), primals, tangents);
    }

    std::array<std::unique_ptr<mlx::core::array>, 2> jvp(
        const UnaryFn *fun,
        const mlx::core::array &primal,
        const mlx::core::array &tangent)
    {
        return mlx_cxx::jvp(make_unary_fn(fun), primal, tangent);
    }

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const MultiaryFn *fun,
        const std::vector<int> &argnums)
    {
        return mlx_cxx::value_and_grad(make_multiary_fn(fun), argnums);
    }

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const MultiaryFn *fun,
        int argnum)
    {
        return mlx_cxx::value_and_grad(make_multiary_fn(fun), argnum);
    }

    std::unique_ptr<CxxSingleInputPairOutputFn> value_and_grad(
        const UnaryFn *fun)
    {
        return mlx_cxx::value_and_grad(make_unary_fn(fun));
    }

    std::unique_ptr<mlx::core::SimpleValueAndGradFn> value_and_grad(
        const MultiInputSingleOutputFn *fun,
        const std::vector<int> &argnums)
    {
        return mlx_cxx::value_and_grad(make_multi_input_single_output_fn(fun), argnums);
    }

    std::unique_ptr<CxxMultiaryFn> grad(
        const MultiInputSingleOutputFn *fun,
        const std::vector<int> &argnums)
    {
        return mlx_cxx::grad(make_multi_input_single_output_fn(fun), argnums);
    }

    std::unique_ptr<CxxMultiaryFn> grad(
        const MultiInputSingleOutputFn *fun,
        int argnum)
    {
        return mlx_cxx::grad(make_multi_input_single_output_fn(fun), argnum);
    }

    std::unique_ptr<CxxUnaryFn> grad(
        const UnaryFn *fun)
    {
        return mlx_cxx::grad(make_unary_fn(fun));
    }

    std::unique_ptr<CxxUnaryFn> vmap(
        const UnaryFn *fun,
        int in_axis,
        int out_axis)
    {
        return mlx_cxx::vmap(make_unary_fn(fun), in_axis, out_axis);
    }

    std::unique_ptr<CxxPairInputSingleOutputFn> vmap(
        const PairInputSingleOutputFn *fun,
        int in_axis_a,
        int in_axis_b,
        int out_axis)
    {
        return mlx_cxx::vmap(make_pair_input_single_output_fn(fun), in_axis_a, in_axis_b, out_axis);
    }

    std::unique_ptr<CxxMultiaryFn> vmap(
        const MultiaryFn *fun,
        const std::vector<int> &in_axes,
        const std::vector<int> &out_axes)
    {
        return mlx_cxx::vmap(make_multiary_fn(fun), in_axes, out_axes);
    }

    std::unique_ptr<CxxMultiaryFn> custom_vjp(
        const MultiaryFn *fun,
        const VjpFn *fun_vjp)
    {
        auto cxx_fun = make_multiary_fn(fun);
        auto cxx_vjp_fun = make_vjp_fn(fun_vjp);
        return std::make_unique<CxxMultiaryFn>(mlx::core::custom_vjp(cxx_fun, cxx_vjp_fun));
    }

    std::unique_ptr<CxxMultiaryFn> checkpoint(
        const MultiaryFn *fun)
    {
        auto cxx_fun = make_multiary_fn(fun);
        return std::make_unique<CxxMultiaryFn>(mlx::core::checkpoint(cxx_fun));
    }

    /* -------------------------------------------------------------------------- */

    std::unique_ptr<CxxMultiaryFn> compile(const MultiaryFn *fun)
    {
        auto cxx_fun = make_multiary_fn(fun);
        return std::make_unique<CxxMultiaryFn>(mlx::core::compile(cxx_fun));
    }
}