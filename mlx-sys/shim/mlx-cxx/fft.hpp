#pragma once

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/fft.h"

#include "mlx-cxx/mlx_cxx.hpp"

namespace mlx_cxx
{
    using mlx::core::array;

    std::unique_ptr<array> fftn(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<array> fftn(
        const array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<array> fftn(
        const array &a,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<array> ifftn(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<array> ifftn(
        const array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<array> ifftn(
        const array &a,
        mlx_cxx::StreamOrDevice s = {});

    /** Compute the one-dimensional Fourier Transform. */
    inline std::unique_ptr<array> fft(const array &a, int n, int axis, mlx_cxx::StreamOrDevice s = {})
    {
        return fftn(a, {n}, {axis}, s);
    }
    inline std::unique_ptr<array> fft(const array &a, int axis = -1, mlx_cxx::StreamOrDevice s = {})
    {
        return fftn(a, {axis}, s);
    }
    inline std::unique_ptr<array> fft(const array &a, mlx_cxx::StreamOrDevice s = {})
    {
        return fftn(a, {-1}, s);
    }

    /** Compute the one-dimensional inverse Fourier Transform. */
    inline std::unique_ptr<array> ifft(const array &a, int n, int axis, mlx_cxx::StreamOrDevice s = {})
    {
        return ifftn(a, {n}, {axis}, s);
    }
    inline std::unique_ptr<array> ifft(const array &a, int axis = -1, mlx_cxx::StreamOrDevice s = {})
    {
        return ifftn(a, {axis}, s);
    }
    inline std::unique_ptr<array> ifft(const array &a, mlx_cxx::StreamOrDevice s = {})
    {
        return ifftn(a, {-1}, s);
    }

    /** Compute the two-dimensional Fourier Transform. */
    inline std::unique_ptr<array> fft2(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {})
    {
        return fftn(a, n, axes, s);
    }
    inline std::unique_ptr<array> fft2(
        const array &a,
        const std::vector<int> &axes = {-2, -1},
        mlx_cxx::StreamOrDevice s = {})
    {
        return fftn(a, axes, s);
    }
    inline std::unique_ptr<array> fft2(
        const array &a,
        mlx_cxx::StreamOrDevice s = {})
    {
        return fftn(a, {-2, -1}, s);
    }

    /** Compute the two-dimensional inverse Fourier Transform. */
    inline std::unique_ptr<array> ifft2(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {})
    {
        return ifftn(a, n, axes, s);
    }
    inline std::unique_ptr<array> ifft2(
        const array &a,
        const std::vector<int> &axes = {-2, -1},
        mlx_cxx::StreamOrDevice s = {})
    {
        return ifftn(a, axes, s);
    }
    inline std::unique_ptr<array> ifft2(
        const array &a,
        mlx_cxx::StreamOrDevice s = {})
    {
        return ifftn(a, {-2, -1}, s);
    }

    /** Compute the n-dimensional Fourier Transform on a real input. */
    std::unique_ptr<array> rfftn(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<array> rfftn(
        const array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<array> rfftn(const array &a, mlx_cxx::StreamOrDevice s = {});

    /** Compute the n-dimensional inverse of `rfftn`. */
    std::unique_ptr<array> irfftn(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<array> irfftn(
        const array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<array> irfftn(const array &a, mlx_cxx::StreamOrDevice s = {});

    /** Compute the one-dimensional Fourier Transform on a real input. */
    inline std::unique_ptr<array> rfft(const array &a, int n, int axis, mlx_cxx::StreamOrDevice s = {})
    {
        return rfftn(a, {n}, {axis}, s);
    }
    inline std::unique_ptr<array> rfft(const array &a, int axis = -1, mlx_cxx::StreamOrDevice s = {})
    {
        return rfftn(a, {axis}, s);
    }
    inline std::unique_ptr<array> rfft(const array &a, mlx_cxx::StreamOrDevice s = {})
    {
        return rfftn(a, {-1}, s);
    }

    /** Compute the one-dimensional inverse of `rfft`. */
    inline std::unique_ptr<array> irfft(const array &a, int n, int axis, mlx_cxx::StreamOrDevice s = {})
    {
        return irfftn(a, {n}, {axis}, s);
    }
    inline std::unique_ptr<array> irfft(const array &a, int axis = -1, mlx_cxx::StreamOrDevice s = {})
    {
        return irfftn(a, {axis}, s);
    }
    inline std::unique_ptr<array> irfft(const array &a, mlx_cxx::StreamOrDevice s = {})
    {
        return irfftn(a, {-1}, s);
    }

    /** Compute the two-dimensional Fourier Transform on a real input. */
    inline std::unique_ptr<array> rfft2(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {})
    {
        return rfftn(a, n, axes, s);
    }
    inline std::unique_ptr<array> rfft2(
        const array &a,
        const std::vector<int> &axes = {-2, -1},
        mlx_cxx::StreamOrDevice s = {})
    {
        return rfftn(a, axes, s);
    }
    inline std::unique_ptr<array> rfft2(
        const array &a,
        mlx_cxx::StreamOrDevice s = {})
    {
        return rfftn(a, {-2, -1}, s);
    }

    /** Compute the two-dimensional inverse of `rfft2`. */
    inline std::unique_ptr<array> irfft2(
        const array &a,
        const std::vector<int> &n,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {})
    {
        return irfftn(a, n, axes, s);
    }
    inline std::unique_ptr<array> irfft2(
        const array &a,
        const std::vector<int> &axes = {-2, -1},
        mlx_cxx::StreamOrDevice s = {})
    {
        return irfftn(a, axes, s);
    }
    inline std::unique_ptr<array> irfft2(
        const array &a,
        mlx_cxx::StreamOrDevice s = {})
    {
        return irfftn(a, {-2, -1}, s);
    }
}