#include "mlx/transforms.h"
#include "mlx-cxx/transforms.hpp"

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
        // const std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array> &)> &fun,
        const MultiaryFn &fun,
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
}