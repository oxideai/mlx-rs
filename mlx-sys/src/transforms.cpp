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

    // std::array<std::unique_ptr<std::vector<mlx::core::array>>, 2> vjp(
    //     const MultiaryFn &fun,
    //     rust::Slice<const std::unique_ptr<mlx::core::array>> primals,
    //     rust::Slice<const std::unique_ptr<mlx::core::array>> cotangents)
    // {
    //     // Wrap the MultiaryFn in a MultiaryCxxFn
    //     auto cxx_fun = [fun=std::move(fun)](const std::vector<mlx::core::array> &args) {
    //         auto ptr = mlx_cxx::execute_multiary_fn(fun, args);
    //         return *ptr;
    //     };
    // }

    std::array<std::unique_ptr<mlx::core::array>, 2> vjp(
        const UnaryFn *fun,
        const mlx::core::array &primal,
        const mlx::core::array &cotangent)
    {
        auto result = mlx::core::vjp(make_unary_fn(fun), primal, cotangent);
        auto first = std::make_unique<mlx::core::array>(result.first);
        auto second = std::make_unique<mlx::core::array>(result.second);

        return {std::move(first), std::move(second)};
    }
}