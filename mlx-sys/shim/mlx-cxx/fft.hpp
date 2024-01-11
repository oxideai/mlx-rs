#pragma once

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/fft.h"

namespace mlx_cxx {
    // We would like to map `StreamOrDevice` from c++ to something like below
    // in Rust:
    // ```rust
    // enum StreamOrDevice {
    //     Default, // default value
    //     Stream(Stream),
    //     Device(Device),
    // }
    // ```

    std::unique_ptr<mlx::core::array> fftn_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> fftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> fftn_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> fftn_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> fftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> fftn_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> fftn_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> fftn_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> fftn_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> ifftn_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> ifftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> ifftn_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> ifftn_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> ifftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> ifftn_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> ifftn_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> ifftn_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> ifftn_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> fft_default(
        const mlx::core::array& a,
        int n,
        int axis
    );

    std::unique_ptr<mlx::core::array> fft_stream(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> fft_device(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> fft_default(
        const mlx::core::array& a,
        int axis
    );

    std::unique_ptr<mlx::core::array> fft_stream(
        const mlx::core::array& a,
        int axis,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> fft_device(
        const mlx::core::array& a,
        int axis,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> fft_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> fft_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> fft_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> ifft_default(
        const mlx::core::array& a,
        int n,
        int axis
    );

    std::unique_ptr<mlx::core::array> ifft_stream(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> ifft_device(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> ifft_default(
        const mlx::core::array& a,
        int axis
    );

    std::unique_ptr<mlx::core::array> ifft_stream(
        const mlx::core::array& a,
        int axis,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> ifft_device(
        const mlx::core::array& a,
        int axis,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> ifft_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> ifft_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> ifft_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> fft2_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> fft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> fft2_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> fft2_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> fft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> fft2_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> fft2_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> fft2_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> fft2_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> ifft2_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> ifft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> ifft2_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> ifft2_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> ifft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> ifft2_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> ifft2_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> ifft2_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> ifft2_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> rfftn_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> rfftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> rfftn_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> rfftn_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> rfftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> rfftn_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> rfftn_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> rfftn_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> rfftn_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> irfftn_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> irfftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> irfftn_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> irfftn_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> irfftn_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> irfftn_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> irfftn_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> irfftn_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> irfftn_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> rfft_default(
        const mlx::core::array& a,
        int n,
        int axis
    );

    std::unique_ptr<mlx::core::array> rfft_stream(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> rfft_device(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> rfft_default(
        const mlx::core::array& a,
        int axis
    );

    std::unique_ptr<mlx::core::array> rfft_stream(
        const mlx::core::array& a,
        int axis,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> rfft_device(
        const mlx::core::array& a,
        int axis,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> rfft_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> rfft_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> rfft_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> irfft_default(
        const mlx::core::array& a,
        int n,
        int axis
    );

    std::unique_ptr<mlx::core::array> irfft_stream(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> irfft_device(
        const mlx::core::array& a,
        int n,
        int axis,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> irfft_default(
        const mlx::core::array& a,
        int axis
    );

    std::unique_ptr<mlx::core::array> irfft_stream(
        const mlx::core::array& a,
        int axis,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> irfft_device(
        const mlx::core::array& a,
        int axis,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> irfft_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> irfft_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> irfft_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );
    
    std::unique_ptr<mlx::core::array> rfft2_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> rfft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> rfft2_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> rfft2_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> rfft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> rfft2_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> rfft2_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> rfft2_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> rfft2_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> irfft2_default(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> irfft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> irfft2_device(
        const mlx::core::array& a,
        const std::vector<int>& n,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> irfft2_default(
        const mlx::core::array& a,
        const std::vector<int>& axes
    );

    std::unique_ptr<mlx::core::array> irfft2_stream(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> irfft2_device(
        const mlx::core::array& a,
        const std::vector<int>& axes,
        mlx::core::Device d
    );

    std::unique_ptr<mlx::core::array> irfft2_default(
        const mlx::core::array& a
    );

    std::unique_ptr<mlx::core::array> irfft2_stream(
        const mlx::core::array& a,
        mlx::core::Stream s
    );

    std::unique_ptr<mlx::core::array> irfft2_device(
        const mlx::core::array& a,
        mlx::core::Device d
    );
}