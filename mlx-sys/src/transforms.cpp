#include "mlx/transforms.h"

#include "mlx-cxx/transforms.hpp"

#include "mlx-sys/src/function.rs.h"

#include "rust/cxx.h"

namespace mlx_cxx
{
    // int execute_callback(const mlx_cxx::DynFn &f, int args)
    // {
    //     return mlx_cxx::execute_dyn_fn(f, args);
    // }

    // TODO: should this use a mutable Slice?
    void simplify(rust::Slice<const std::unique_ptr<mlx::core::array>> outputs)
    {
        auto outputs_vec = std::vector<mlx::core::array>{};
        for (auto &output : outputs)
        {
            outputs_vec.push_back(*output);
        }

        mlx::core::simplify(outputs_vec);
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
        const MultiaryCxxFn &fun,
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
        const UnaryCxxFn &fun,
        const mlx::core::array &primal,
        const mlx::core::array &cotangent)
    {
        auto result = mlx::core::vjp(fun, primal, cotangent);
        auto first = std::make_unique<mlx::core::array>(result.first);
        auto second = std::make_unique<mlx::core::array>(result.second);

        return {std::move(first), std::move(second)};
    }

    std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> jvp(
        const MultiaryCxxFn &fun,
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
        const UnaryCxxFn &fun,
        const mlx::core::array &primal,
        const mlx::core::array &tangent)
    {
        auto result = mlx::core::jvp(fun, primal, tangent);
        auto first = std::make_unique<mlx::core::array>(result.first);
        auto second = std::make_unique<mlx::core::array>(result.second);

        return {std::move(first), std::move(second)};
    }

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const MultiaryCxxFn &fun,
        const std::vector<int> &argnums)
    {
        return std::make_unique<mlx::core::ValueAndGradFn>(mlx::core::value_and_grad(fun, argnums));
    }

    std::unique_ptr<mlx::core::ValueAndGradFn> value_and_grad(
        const MultiaryCxxFn &fun,
        int argnum)
    {
        return std::make_unique<mlx::core::ValueAndGradFn>(mlx::core::value_and_grad(fun, argnum));
    }

    std::unique_ptr<SipoCxxFn> value_and_grad(
        const UnaryCxxFn &fun)
    {
        return std::make_unique<SipoCxxFn>(mlx::core::value_and_grad(fun));
    }

    std::unique_ptr<mlx::core::SimpleValueAndGradFn> value_and_grad(
        const MisoCxxFn &fun,
        const std::vector<int> &argnums)
    {
        return std::make_unique<mlx::core::SimpleValueAndGradFn>(mlx::core::value_and_grad(fun, argnums));
    }

    std::unique_ptr<MultiaryCxxFn> grad(
        const MisoCxxFn &fun,
        const std::vector<int> &argnums)
    {
        return std::make_unique<MultiaryCxxFn>(mlx::core::grad(fun, argnums));
    }

    std::unique_ptr<MultiaryCxxFn> grad(
        const MisoCxxFn &fun,
        int argnum)
    {
        return std::make_unique<MultiaryCxxFn>(mlx::core::grad(fun, argnum));
    }

    std::unique_ptr<UnaryCxxFn> grad(
        const UnaryCxxFn &fun)
    {
        return std::make_unique<UnaryCxxFn>(mlx::core::grad(fun));
    }

    std::unique_ptr<UnaryCxxFn> vmap(
        const UnaryCxxFn &fun,
        int in_axis,
        int out_axis)
    {
        return std::make_unique<UnaryCxxFn>(mlx::core::vmap(fun, in_axis, out_axis));
    }

    std::unique_ptr<PisoCxxFn> vmap(
        const PisoCxxFn &fun,
        int in_axis_a,
        int in_axis_b,
        int out_axis)
    {
        return std::make_unique<PisoCxxFn>(mlx::core::vmap(fun, in_axis_a, in_axis_b, out_axis));
    }

    std::unique_ptr<MultiaryCxxFn> vmap(
        const MultiaryCxxFn &fun,
        const std::vector<int> &in_axes,
        const std::vector<int> &out_axes)
    {
        return std::make_unique<MultiaryCxxFn>(mlx::core::vmap(fun, in_axes, out_axes));
    }

    /* -------------------------------------------------------------------------- */
    /*                     Bindings that accept rust funcionts                    */
    /* -------------------------------------------------------------------------- */


    int accept_rust_unary_fn(const mlx_cxx::UnaryFn &f)
    {
        return 1;
    }

    UnaryCxxFn make_unary_fn(const UnaryFn *f)
    {
        return [fun = std::move(f)](const mlx::core::array &arg)
        {
            auto ptr = mlx_cxx::execute_unary_fn(*fun, arg);
            return *ptr;
        };
    }

    MultiaryCxxFn make_multiary_fn(const MultiaryFn *f)
    {
        return [fun = std::move(f)](const std::vector<mlx::core::array> &args)
        {
            auto ptr = mlx_cxx::execute_multiary_fn(*fun, args);
            return *ptr;
        };
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
}