#include "mlx-cxx/fft.hpp"

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/fft.h"

namespace mlx_cxx {
    std::unique_ptr<mlx::core::array> fftn_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fftn(a, n, axes));
    }

    std::unique_ptr<mlx::core::array> fftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fftn(a, n, axes, s));
    }

    std::unique_ptr<mlx::core::array> fftn_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fftn(a, n, axes, d));
    }

    std::unique_ptr<mlx::core::array> fftn_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fftn(a, axes));
    }

    std::unique_ptr<mlx::core::array> fftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fftn(a, axes, s));
    }

    std::unique_ptr<mlx::core::array> fftn_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fftn(a, axes, d));
    }

    std::unique_ptr<mlx::core::array> fftn_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fftn(a));
    }

    std::unique_ptr<mlx::core::array> fftn_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fftn(a, s));
    }

    std::unique_ptr<mlx::core::array> fftn_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fftn(a, d));
    }

    std::unique_ptr<mlx::core::array> ifftn_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifftn(a, n, axes));
    }

    std::unique_ptr<mlx::core::array> ifftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifftn(a, n, axes, s));
    }

    std::unique_ptr<mlx::core::array> ifftn_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifftn(a, n, axes, d));
    }

    std::unique_ptr<mlx::core::array> ifftn_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifftn(a, axes));
    }

    std::unique_ptr<mlx::core::array> ifftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifftn(a, axes, s));
    }

    std::unique_ptr<mlx::core::array> ifftn_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifftn(a, axes, d));
    }

    std::unique_ptr<mlx::core::array> ifftn_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifftn(a));
    }

    std::unique_ptr<mlx::core::array> ifftn_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifftn(a, s));
    }

    std::unique_ptr<mlx::core::array> ifftn_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifftn(a, d));
    }

    std::unique_ptr<mlx::core::array> fft_default(
        const mlx::core::array& a,
        int n,
        int axis
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft(a, n, axis));
    }

    std::unique_ptr<mlx::core::array> fft_stream(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft(a, n, axis, s));
    }

    std::unique_ptr<mlx::core::array> fft_device(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft(a, n, axis, d));
    }

    std::unique_ptr<mlx::core::array> fft_default(
        const mlx::core::array& a,
        int axis
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft(a, axis));
    }

    std::unique_ptr<mlx::core::array> fft_stream(
        const mlx::core::array& a,
        int axis,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft(a, axis, s));
    }

    std::unique_ptr<mlx::core::array> fft_device(
        const mlx::core::array& a,
        int axis,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft(a, axis, d));
    }

    std::unique_ptr<mlx::core::array> fft_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft(a));
    }

    std::unique_ptr<mlx::core::array> fft_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft(a, -1, s));
    }

    std::unique_ptr<mlx::core::array> fft_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft(a, -1, d));
    }

    std::unique_ptr<mlx::core::array> ifft_default(
        const mlx::core::array& a,
        int n,
        int axis
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft(a, n, axis));
    }

    std::unique_ptr<mlx::core::array> ifft_stream(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft(a, n, axis, s));
    }

    std::unique_ptr<mlx::core::array> ifft_device(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft(a, n, axis, d));
    }

    std::unique_ptr<mlx::core::array> ifft_default(
        const mlx::core::array& a,
        int axis
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft(a, axis));
    }

    std::unique_ptr<mlx::core::array> ifft_stream(
        const mlx::core::array& a,
        int axis,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft(a, axis, s));
    }

    std::unique_ptr<mlx::core::array> ifft_device(
        const mlx::core::array& a,
        int axis,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft(a, axis, d));
    }

    std::unique_ptr<mlx::core::array> ifft_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft(a));
    }

    std::unique_ptr<mlx::core::array> ifft_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft(a, -1, s));
    }

    std::unique_ptr<mlx::core::array> ifft_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft(a, -1, d));
    }

    std::unique_ptr<mlx::core::array> fft2_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft2(a, n, axes));
    }

    std::unique_ptr<mlx::core::array> fft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft2(a, n, axes, s));
    }

    std::unique_ptr<mlx::core::array> fft2_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft2(a, n, axes, d));
    }

    std::unique_ptr<mlx::core::array> fft2_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft2(a, axes));
    }

    std::unique_ptr<mlx::core::array> fft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft2(a, axes, s));
    }

    std::unique_ptr<mlx::core::array> fft2_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft2(a, axes, d));
    }

    std::unique_ptr<mlx::core::array> fft2_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft2(a));
    }

    std::unique_ptr<mlx::core::array> fft2_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft2(a, {-2, -1}, s));
    }

    std::unique_ptr<mlx::core::array> fft2_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::fft2(a, {-2, -1}, d));
    }

    std::unique_ptr<mlx::core::array> ifft2_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft2(a, n, axes));
    }

    std::unique_ptr<mlx::core::array> ifft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft2(a, n, axes, s));
    }

    std::unique_ptr<mlx::core::array> ifft2_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft2(a, n, axes, d));
    }

    std::unique_ptr<mlx::core::array> ifft2_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft2(a, axes));
    }

    std::unique_ptr<mlx::core::array> ifft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft2(a, axes, s));
    }

    std::unique_ptr<mlx::core::array> ifft2_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft2(a, axes, d));
    }

    std::unique_ptr<mlx::core::array> ifft2_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft2(a));
    }

    std::unique_ptr<mlx::core::array> ifft2_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft2(a, {-2, -1}, s));
    }

    std::unique_ptr<mlx::core::array> ifft2_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::ifft2(a, {-2, -1}, d));
    }

    std::unique_ptr<mlx::core::array> rfftn_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfftn(a, n, axes));
    }

    std::unique_ptr<mlx::core::array> rfftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfftn(a, n, axes, s));
    }

    std::unique_ptr<mlx::core::array> rfftn_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfftn(a, n, axes, d));
    }

    std::unique_ptr<mlx::core::array> rfftn_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfftn(a, axes));
    }

    std::unique_ptr<mlx::core::array> rfftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfftn(a, axes, s));
    }

    std::unique_ptr<mlx::core::array> rfftn_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfftn(a, axes, d));
    }

    std::unique_ptr<mlx::core::array> rfftn_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfftn(a));
    }

    std::unique_ptr<mlx::core::array> rfftn_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfftn(a, s));
    }

    std::unique_ptr<mlx::core::array> rfftn_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfftn(a, d));
    }

    std::unique_ptr<mlx::core::array> irfftn_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfftn(a, n, axes));
    }

    std::unique_ptr<mlx::core::array> irfftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfftn(a, n, axes, s));
    }

    std::unique_ptr<mlx::core::array> irfftn_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfftn(a, n, axes, d));
    }

    std::unique_ptr<mlx::core::array> irfftn_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfftn(a, axes));
    }

    std::unique_ptr<mlx::core::array> irfftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfftn(a, axes, s));
    }

    std::unique_ptr<mlx::core::array> irfftn_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfftn(a, axes, d));
    }

    std::unique_ptr<mlx::core::array> irfftn_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfftn(a));
    }

    std::unique_ptr<mlx::core::array> irfftn_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfftn(a, s));
    }

    std::unique_ptr<mlx::core::array> irfftn_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfftn(a, d));
    }
    
    std::unique_ptr<mlx::core::array> rfft_default(
        const mlx::core::array& a,
        int n,
        int axis
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft(a, n, axis));
    }

    std::unique_ptr<mlx::core::array> rfft_stream(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft(a, n, axis, s));
    }

    std::unique_ptr<mlx::core::array> rfft_device(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft(a, n, axis, d));
    }

    std::unique_ptr<mlx::core::array> rfft_default(
        const mlx::core::array& a,
        int axis
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft(a, axis));
    }

    std::unique_ptr<mlx::core::array> rfft_stream(
        const mlx::core::array& a,
        int axis,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft(a, axis, s));
    }

    std::unique_ptr<mlx::core::array> rfft_device(
        const mlx::core::array& a,
        int axis,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft(a, axis, d));
    }

    std::unique_ptr<mlx::core::array> rfft_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft(a));
    }

    std::unique_ptr<mlx::core::array> rfft_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft(a, -1, s));
    }

    std::unique_ptr<mlx::core::array> rfft_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft(a, -1, d));
    }

    std::unique_ptr<mlx::core::array> irfft_default(
        const mlx::core::array& a,
        int n,
        int axis
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft(a, n, axis));
    }

    std::unique_ptr<mlx::core::array> irfft_stream(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft(a, n, axis, s));
    }

    std::unique_ptr<mlx::core::array> irfft_device(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft(a, n, axis, d));
    }

    std::unique_ptr<mlx::core::array> irfft_default(
        const mlx::core::array& a,
        int axis
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft(a, axis));
    }

    std::unique_ptr<mlx::core::array> irfft_stream(
        const mlx::core::array& a,
        int axis,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft(a, axis, s));
    }

    std::unique_ptr<mlx::core::array> irfft_device(
        const mlx::core::array& a,
        int axis,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft(a, axis, d));
    }

    std::unique_ptr<mlx::core::array> irfft_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft(a));
    }

    std::unique_ptr<mlx::core::array> irfft_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft(a, -1, s));
    }

    std::unique_ptr<mlx::core::array> irfft_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft(a, -1, d));
    }

    std::unique_ptr<mlx::core::array> rfft2_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft2(a, n, axes));
    }

    std::unique_ptr<mlx::core::array> rfft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft2(a, n, axes, s));
    }

    std::unique_ptr<mlx::core::array> rfft2_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft2(a, n, axes, d));
    }

    std::unique_ptr<mlx::core::array> rfft2_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft2(a, axes));
    }

    std::unique_ptr<mlx::core::array> rfft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft2(a, axes, s));
    }

    std::unique_ptr<mlx::core::array> rfft2_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft2(a, axes, d));
    }

    std::unique_ptr<mlx::core::array> rfft2_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft2(a));
    }

    std::unique_ptr<mlx::core::array> rfft2_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft2(a, {-2, -1}, s));
    }

    std::unique_ptr<mlx::core::array> rfft2_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::rfft2(a, {-2, -1}, d));
    }

    std::unique_ptr<mlx::core::array> irfft2_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft2(a, n, axes));
    }

    std::unique_ptr<mlx::core::array> irfft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft2(a, n, axes, s));
    }

    std::unique_ptr<mlx::core::array> irfft2_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft2(a, n, axes, d));
    }

    std::unique_ptr<mlx::core::array> irfft2_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft2(a, axes));
    }

    std::unique_ptr<mlx::core::array> irfft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft2(a, axes, s));
    }

    std::unique_ptr<mlx::core::array> irfft2_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft2(a, axes, d));
    }

    std::unique_ptr<mlx::core::array> irfft2_default(
        const mlx::core::array& a
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft2(a));
    }

    std::unique_ptr<mlx::core::array> irfft2_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft2(a, {-2, -1}, s));
    }

    std::unique_ptr<mlx::core::array> irfft2_device(
        const mlx::core::array& a,
        mlx::core::Device d
    ) {
        return std::make_unique<mlx::core::array>(mlx::core::fft::irfft2(a, {-2, -1}, d));
    }
}