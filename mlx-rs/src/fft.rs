use cxx::CxxVector;
use mlx_sys::{array::ffi::array, utils::StreamOrDevice};

use crate::array::Array;

pub fn fftn_shape_axes(
    a: impl AsRef<array>,
    n: impl AsRef<CxxVector<i32>>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let n = n.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::fftn_shape_axes(a, n, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn fftn_axes(
    a: impl AsRef<array>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::fftn_axes(a, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn fftn(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::fftn(a, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn ifftn_shape_axes(
    a: impl AsRef<array>,
    n: impl AsRef<CxxVector<i32>>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let n = n.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::ifftn_shape_axes(a, n, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn ifftn_axes(
    a: impl AsRef<array>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::ifftn_axes(a, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn ifftn(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::ifftn(a, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn fft_shape_axis(
    a: impl AsRef<array>,
    n: i32,
    axis: i32,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::fft_shape_axis(a, n, axis, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn fft_axis(
    a: impl AsRef<array>,
    axis: i32,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::fft_axis(a, axis, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn fft(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::fft(a, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn ifft_shape_axis(
    a: impl AsRef<array>,
    n: i32,
    axis: i32,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::ifft_shape_axis(a, n, axis, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn ifft_axis(
    a: impl AsRef<array>,
    axis: i32,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::ifft_axis(a, axis, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn ifft(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::ifft(a, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn fft2_shape_axes(
    a: impl AsRef<array>,
    n: impl AsRef<CxxVector<i32>>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let n = n.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::fft2_shape_axes(a, n, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn fft2_axes(
    a: impl AsRef<array>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::fft2_axes(a, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn fft2(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::fft2(a, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn ifft2_shape_axes(
    a: impl AsRef<array>,
    n: impl AsRef<CxxVector<i32>>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let n = n.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::ifft2_shape_axes(a, n, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn ifft2_axes(
    a: impl AsRef<array>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::ifft2_axes(a, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn ifft2(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::ifft2(a, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn rfftn_shape_axes(
    a: impl AsRef<array>,
    n: impl AsRef<CxxVector<i32>>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let n = n.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::rfftn_shape_axes(a, n, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn rfftn_axes(
    a: impl AsRef<array>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::rfftn_axes(a, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn rfftn(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::rfftn(a, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn irfftn_shape_axes(
    a: impl AsRef<array>,
    n: impl AsRef<CxxVector<i32>>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let n = n.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::irfftn_shape_axes(a, n, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn irfftn_axes(
    a: impl AsRef<array>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::irfftn_axes(a, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn irfftn(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::irfftn(a, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn rfft_shape_axis(
    a: impl AsRef<array>,
    n: i32,
    axis: i32,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::rfft_shape_axis(a, n, axis, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn rfft_axis(
    a: impl AsRef<array>,
    axis: i32,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::rfft_axis(a, axis, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn rfft(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::rfft(a, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn irfft_shape_axis(
    a: impl AsRef<array>,
    n: i32,
    axis: i32,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::irfft_shape_axis(a, n, axis, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn irfft_axis(
    a: impl AsRef<array>,
    axis: i32,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::irfft_axis(a, axis, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn irfft(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::irfft(a, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn rfft2_shape_axes(
    a: impl AsRef<array>,
    n: impl AsRef<CxxVector<i32>>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let n = n.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::rfft2_shape_axes(a, n, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn rfft2_axes(
    a: impl AsRef<array>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::rfft2_axes(a, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn rfft2(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::rfft2(a, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn irfft2_shape_axes(
    a: impl AsRef<array>,
    n: impl AsRef<CxxVector<i32>>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let n = n.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::irfft2_shape_axes(a, n, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn irfft2_axes(
    a: impl AsRef<array>,
    axes: impl AsRef<CxxVector<i32>>,
    s: StreamOrDevice,
) -> Array {
    let a = a.as_ref();
    let axes = axes.as_ref();
    let res = mlx_sys::fft::ffi::irfft2_axes(a, axes, s);
    Array {
        inner: res.unwrap(),
    }
}

pub fn irfft2(a: impl AsRef<array>, s: StreamOrDevice) -> Array {
    let a = a.as_ref();
    let res = mlx_sys::fft::ffi::irfft2(a, s);
    Array {
        inner: res.unwrap(),
    }
}
