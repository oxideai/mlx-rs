#include "mlx-cxx/mlx_cxx.hpp"
#include "mlx-cxx/fft.hpp"
#include "mlx-cxx/utils.hpp"

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/fft.h"

namespace mlx_cxx
{
    std::unique_ptr<array> fftn(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::fftn(a, n, axes, s.to_variant()));
    }

    std::unique_ptr<array> fftn(
        const array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::fftn(a, axes, s.to_variant()));
    }

    std::unique_ptr<array> fftn(
        const array &a,
        mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::fftn(a, s.to_variant()));
    }

    std::unique_ptr<array> ifftn(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::ifftn(a, n, axes, s.to_variant()));
    }

    std::unique_ptr<array> ifftn(
        const array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::ifftn(a, axes, s.to_variant()));
    }

    std::unique_ptr<array> ifftn(
        const array &a,
        mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::ifftn(a, s.to_variant()));
    }

    /** Compute the n-dimensional Fourier Transform on a real input. */
    std::unique_ptr<array> rfftn(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::rfftn(a, n, axes, s.to_variant()));
    }

    std::unique_ptr<array> rfftn(
        const array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::rfftn(a, axes, s.to_variant()));
    }

    std::unique_ptr<array> rfftn(const array &a, mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::rfftn(a, s.to_variant()));
    }

    /** Compute the n-dimensional inverse of `rfftn`. */
    std::unique_ptr<array> irfftn(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::irfftn(a, n, axes, s.to_variant()));
    }

    std::unique_ptr<array> irfftn(
        const array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::irfftn(a, axes, s.to_variant()));
    }

    std::unique_ptr<array> irfftn(const array &a, mlx_cxx::StreamOrDevice s)
    {
        return std::make_unique<array>(mlx::core::fft::irfftn(a, s.to_variant()));
    }

}