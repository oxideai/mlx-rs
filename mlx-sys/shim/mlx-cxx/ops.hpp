#pragma once

#include "mlx/ops.h"
#include "mlx/array.h"
#include "mlx/dtype.h"
#include "mlx/io/load.h"

#include "mlx-cxx/mlx_cxx.hpp"

#include "rust/cxx.h"

namespace mlx_cxx
{
    using OptionalArray = mlx_cxx::Optional<std::unique_ptr<mlx::core::array>>;

    std::optional<mlx::core::array> to_std_optional(const OptionalArray &opt);

    /** Creation operations */

    mlx::core::Stream to_stream(mlx_cxx::StreamOrDevice s)
    {
        return mlx::core::to_stream(s.to_variant());
    }

    /**
     * A 1D std::unique_ptr<mlx::core::array> of numbers starting at `start` (optional),
     * stopping at stop, stepping by `step` (optional). */
    std::unique_ptr<mlx::core::array> arange(
        double start,
        double stop,
        double step,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> arange(double start, double stop, double step, mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> arange(double start, double stop, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> arange(double start, double stop, mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> arange(double stop, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> arange(double stop, mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> arange(int start, int stop, int step, mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> arange(int start, int stop, mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> arange(int stop, mlx_cxx::StreamOrDevice s = {});

    /** A 1D std::unique_ptr<mlx::core::array> of `num` evenly spaced numbers in the range `[start, stop]` */
    std::unique_ptr<mlx::core::array> linspace(
        double start,
        double stop,
        int num,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {});

    /** Convert an std::unique_ptr<mlx::core::array> to the given data type. */
    std::unique_ptr<mlx::core::array> astype(const mlx::core::array &a, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s = {});

    /** Create a view of an std::unique_ptr<mlx::core::array> with the given shape and strides. */
    std::unique_ptr<mlx::core::array> as_strided(
        const mlx::core::array &a,
        std::unique_ptr<std::vector<int>> shape,
        std::unique_ptr<std::vector<size_t>> strides,
        size_t offset,
        mlx_cxx::StreamOrDevice s = {});

    /** Copy another array. */
    std::unique_ptr<mlx::core::array> copy(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Fill an std::unique_ptr<mlx::core::array> of the given shape with the given value(s). */
    std::unique_ptr<mlx::core::array> full(
        const std::vector<int> &shape,
        const mlx::core::array &vals,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> full(
        const std::vector<int> &shape,
        const mlx::core::array &vals,
        mlx_cxx::StreamOrDevice s = {});

    // template <typename T>
    // std::unique_ptr<mlx::core::array> full(
    //     const std::vector<int> &shape,
    //     T val,
    //     mlx::core::Dtype dtype,
    //     mlx_cxx::StreamOrDevice s = {})
    // {
    //     return full(shape, array(val, dtype), to_stream(s));
    // }

    std::unique_ptr<mlx::core::array> full_bool_val_dtype(
        const std::vector<int> &shape,
        bool val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_uint8_val_dtype(
        const std::vector<int> &shape,
        uint8_t val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_uint16_val_dtype(
        const std::vector<int> &shape,
        uint16_t val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_uint32_val_dtype(
        const std::vector<int> &shape,
        uint32_t val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_uint64_val_dtype(
        const std::vector<int> &shape,
        uint64_t val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_int8_val_dtype(
        const std::vector<int> &shape,
        int8_t val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_int16_val_dtype(
        const std::vector<int> &shape,
        int16_t val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_int32_val_dtype(
        const std::vector<int> &shape,
        int32_t val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_int64_val_dtype(
        const std::vector<int> &shape,
        int64_t val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_float16_val_dtype(
        const std::vector<int> &shape,
        mlx::core::float16_t val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_float32_val_dtype(
        const std::vector<int> &shape,
        float val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_bfloat16_val_dtype(
        const std::vector<int> &shape,
        mlx::core::bfloat16_t val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_complex64_val_dtype(
        const std::vector<int> &shape,
        mlx::core::complex64_t val,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    // template <typename T>
    // std::unique_ptr<mlx::core::array> full(const std::vector<int> &shape, T val, mlx_cxx::StreamOrDevice s = {})
    // {
    //     return full(shape, array(val), to_stream(s));
    // }

    std::unique_ptr<mlx::core::array> full_bool_val(
        const std::vector<int> &shape,
        bool val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_uint8_val(
        const std::vector<int> &shape,
        uint8_t val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_uint16_val(
        const std::vector<int> &shape,
        uint16_t val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_uint32_val(
        const std::vector<int> &shape,
        uint32_t val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_uint64_val(
        const std::vector<int> &shape,
        uint64_t val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_int8_val(
        const std::vector<int> &shape,
        int8_t val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_int16_val(
        const std::vector<int> &shape,
        int16_t val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_int32_val(
        const std::vector<int> &shape,
        int32_t val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_int64_val(
        const std::vector<int> &shape,
        int64_t val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_float16_val(
        const std::vector<int> &shape,
        mlx::core::float16_t val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_float32_val(
        const std::vector<int> &shape,
        float val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_bfloat16_val(
        const std::vector<int> &shape,
        mlx::core::bfloat16_t val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> full_complex64_val(
        const std::vector<int> &shape,
        mlx::core::complex64_t val,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::full(shape, val, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Fill an std::unique_ptr<mlx::core::array> of the given shape with zeros. */
    std::unique_ptr<mlx::core::array> zeros(const std::vector<int> &shape, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> zeros(const std::vector<int> &shape, mlx_cxx::StreamOrDevice s = {})
    {
        return zeros(shape, mlx::core::float32, s);
    }
    std::unique_ptr<mlx::core::array> zeros_like(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Fill an std::unique_ptr<mlx::core::array> of the given shape with ones. */
    std::unique_ptr<mlx::core::array> ones(const std::vector<int> &shape, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> ones(const std::vector<int> &shape, mlx_cxx::StreamOrDevice s = {})
    {
        return ones(shape, mlx::core::float32, s);
    }
    std::unique_ptr<mlx::core::array> ones_like(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Fill an std::unique_ptr<mlx::core::array> of the given shape (n,m) with ones in the specified diagonal
     * k, and zeros everywhere else. */
    std::unique_ptr<mlx::core::array> eye(int n, int m, int k, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> eye(int n, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s = {})
    {
        return eye(n, n, 0, dtype, s);
    }
    inline std::unique_ptr<mlx::core::array> eye(int n, int m, mlx_cxx::StreamOrDevice s = {})
    {
        return eye(n, m, 0, mlx::core::float32, s);
    }
    inline std::unique_ptr<mlx::core::array> eye(int n, int m, int k, mlx_cxx::StreamOrDevice s = {})
    {
        return eye(n, m, k, mlx::core::float32, s);
    }
    inline std::unique_ptr<mlx::core::array> eye(int n, mlx_cxx::StreamOrDevice s = {})
    {
        return eye(n, n, 0, mlx::core::float32, s);
    }

    /** Create a square matrix of shape (n,n) of zeros, and ones in the major
     * diagonal. */
    std::unique_ptr<mlx::core::array> identity(int n, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> identity(int n, mlx_cxx::StreamOrDevice s = {})
    {
        return identity(n, mlx::core::float32, s);
    }

    std::unique_ptr<mlx::core::array> tri(int n, int m, int k, mlx::core::Dtype type, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> tri(int n, mlx::core::Dtype type, mlx_cxx::StreamOrDevice s = {})
    {
        return tri(n, n, 0, type, s);
    }

    std::unique_ptr<mlx::core::array> tril(std::unique_ptr<mlx::core::array> x, int k=0, mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> triu(std::unique_ptr<mlx::core::array> x, int k=0, mlx_cxx::StreamOrDevice s = {});

    /** std::unique_ptr<mlx::core::array> manipulation */

    /** Reshape an std::unique_ptr<mlx::core::array> to the given shape. */
    std::unique_ptr<mlx::core::array> reshape(const mlx::core::array &a, std::unique_ptr<std::vector<int>> shape, mlx_cxx::StreamOrDevice s = {});

    /** Flatten the dimensions in the range `[start_axis, end_axis]` . */
    std::unique_ptr<mlx::core::array> flatten(
        const mlx::core::array &a,
        int start_axis,
        int end_axis = -1,
        mlx_cxx::StreamOrDevice s = {});

    /** Flatten the std::unique_ptr<mlx::core::array> to 1D. */
    std::unique_ptr<mlx::core::array> flatten(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Remove singleton dimensions at the given axes. */
    std::unique_ptr<mlx::core::array> squeeze(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});

    /** Remove singleton dimensions at the given axis. */
    inline std::unique_ptr<mlx::core::array> squeeze(const mlx::core::array &a, int axis, mlx_cxx::StreamOrDevice s = {})
    {
        return squeeze(a, std::vector<int>{axis}, s);
    }

    /** Remove all singleton dimensions. */
    std::unique_ptr<mlx::core::array> squeeze(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Add a singleton dimension at the given axes. */
    std::unique_ptr<mlx::core::array> expand_dims(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});

    /** Add a singleton dimension at the given axis. */
    inline std::unique_ptr<mlx::core::array> expand_dims(const mlx::core::array &a, int axis, mlx_cxx::StreamOrDevice s = {})
    {
        return expand_dims(a, std::vector<int>{axis}, s);
    }

    /** Slice an array. */
    std::unique_ptr<mlx::core::array> slice(
        const mlx::core::array &a,
        std::unique_ptr<std::vector<int>> start,
        std::unique_ptr<std::vector<int>> stop,
        std::unique_ptr<std::vector<int>> strides,
        mlx_cxx::StreamOrDevice s = {});

    /** Slice an std::unique_ptr<mlx::core::array> with a stride of 1 in each dimension. */
    std::unique_ptr<mlx::core::array> slice(
        const mlx::core::array &a,
        const std::vector<int> &start,
        const std::vector<int> &stop,
        mlx_cxx::StreamOrDevice s = {});

    /** Split an std::unique_ptr<mlx::core::array> into sub-arrays along a given axis. */
    std::unique_ptr<std::vector<mlx::core::array>>
    split(const mlx::core::array &a, int num_splits, int axis, mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<std::vector<mlx::core::array>> split(const mlx::core::array &a, int num_splits, mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<std::vector<mlx::core::array>> split(
        const mlx::core::array &a,
        const std::vector<int> &indices,
        int axis,
        mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<std::vector<mlx::core::array>>
    split(const mlx::core::array &a, const std::vector<int> &indices, mlx_cxx::StreamOrDevice s = {});

    /**
     * Clip (limit) the values in an array.
     */
    std::unique_ptr<mlx::core::array> clip(
        const mlx::core::array &a,
        const OptionalArray &a_min,
        const OptionalArray &a_max,
        mlx_cxx::StreamOrDevice s = {});

    /** Concatenate arrays along a given axis. */
    std::unique_ptr<mlx::core::array> concatenate(
        const std::vector<mlx::core::array>& arrays,
        int axis,
        mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> concatenate(const std::vector<mlx::core::array>& arrays, mlx_cxx::StreamOrDevice s = {});

    /** Stack arrays along a new axis. */
    std::unique_ptr<mlx::core::array> stack(const std::vector<mlx::core::array>& arrays, int axis, mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> stack(const std::vector<mlx::core::array>& arrays, mlx_cxx::StreamOrDevice s = {});

    /** Repeat an std::unique_ptr<mlx::core::array> along an axis. */
    std::unique_ptr<mlx::core::array> repeat(const mlx::core::array &arr, int repeats, int axis, mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> repeat(const mlx::core::array &arr, int repeats, mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> tile(const mlx::core::array &arr, std::unique_ptr<std::vector<int>> reps, mlx_cxx::StreamOrDevice s = {});

    /** Permutes the dimensions according to the given axes. */
    std::unique_ptr<mlx::core::array> transpose(const mlx::core::array &a, std::unique_ptr<std::vector<int>> axes, mlx_cxx::StreamOrDevice s = {});
    // inline std::unique_ptr<mlx::core::array> transpose(
    //     const mlx::core::array &a,
    //     std::initializer_list<int> axes,
    //     mlx_cxx::StreamOrDevice s = {})
    // {
    //     return transpose(a, std::vector<int>(axes), s);
    // }

    /** Swap two axes of an array. */
    std::unique_ptr<mlx::core::array> swapaxes(const mlx::core::array &a, int axis1, int axis2, mlx_cxx::StreamOrDevice s = {});

    /** Move an axis of an array. */
    std::unique_ptr<mlx::core::array> moveaxis(
        const mlx::core::array &a,
        int source,
        int destination,
        mlx_cxx::StreamOrDevice s = {});

    /** Pad an std::unique_ptr<mlx::core::array> with a constant value */
    std::unique_ptr<mlx::core::array> pad(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        const std::vector<int> &low_pad_size,
        const std::vector<int> &high_pad_size,
        const mlx::core::array &pad_value = mlx::core::array(0),
        mlx_cxx::StreamOrDevice s = {});

    /** Pad an std::unique_ptr<mlx::core::array> with a constant value along all axes */
    // std::unique_ptr<mlx::core::array> pad(
    //     const mlx::core::array &a,
    //     const std::vector<std::pair<int, int>> &pad_width,
    //     const mlx::core::array &pad_value = mlx::core::array(0),
    //     mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> pad(
        const mlx::core::array &a,
        rust::Slice<const std::array<int, 2>> pad_width, // std::pair<int, int>
        const mlx::core::array &pad_value = mlx::core::array(0),
        mlx_cxx::StreamOrDevice s = {});
    // std::unique_ptr<mlx::core::array> pad(
    //     const mlx::core::array &a,
    //     const std::pair<int, int> &pad_width,
    //     const mlx::core::array &pad_value = mlx::core::array(0),
    //     mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> pad(
        const mlx::core::array &a,
        const std::array<int, 2> &pad_width, // std::pair<int, int>
        const mlx::core::array &pad_value = mlx::core::array(0),
        mlx_cxx::StreamOrDevice s = {});
    std::unique_ptr<mlx::core::array> pad(
        const mlx::core::array &a,
        int pad_width,
        const mlx::core::array &pad_value = mlx::core::array(0),
        mlx_cxx::StreamOrDevice s = {});

    /** Permutes the dimensions in reverse order. */
    std::unique_ptr<mlx::core::array> transpose(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Broadcast an std::unique_ptr<mlx::core::array> to a given shape. */
    std::unique_ptr<mlx::core::array> broadcast_to(
        const mlx::core::array &a,
        const std::vector<int> &shape,
        mlx_cxx::StreamOrDevice s = {});

    /** Broadcast a vector of arrays against one another. */
    std::unique_ptr<std::vector<mlx::core::array>> broadcast_arrays(
        const std::vector<mlx::core::array>& inputs,
        mlx_cxx::StreamOrDevice s = {});

    /** Comparison operations */

    /** Returns the bool std::unique_ptr<mlx::core::array> with (a == b) element-wise. */
    std::unique_ptr<mlx::core::array> equal(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // inline std::unique_ptr<mlx::core::array> operator==(const mlx::core::array &a, const mlx::core::array &b)
    // {
    //     return mlx_cxx::equal(a, b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator==(T a, const mlx::core::array &b)
    // {
    //     return equal(array(a), b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator==(const mlx::core::array &a, T b)
    // {
    //     return equal(a, array(b));
    // }

    /** Returns the bool std::unique_ptr<mlx::core::array> with (a != b) element-wise. */
    std::unique_ptr<mlx::core::array> not_equal(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // inline std::unique_ptr<mlx::core::array> operator!=(const mlx::core::array &a, const mlx::core::array &b)
    // {
    //     return mlx_cxx::not_equal(a, b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator!=(T a, const mlx::core::array &b)
    // {
    //     return not_equal(array(a), b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator!=(const mlx::core::array &a, T b)
    // {
    //     return not_equal(a, array(b));
    // }

    /** Returns bool std::unique_ptr<mlx::core::array> with (a > b) element-wise. */
    std::unique_ptr<mlx::core::array> greater(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // inline std::unique_ptr<mlx::core::array> operator>(const mlx::core::array &a, const mlx::core::array &b)
    // {
    //     return mlx_cxx::greater(a, b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator>(T a, const mlx::core::array &b)
    // {
    //     return greater(array(a), b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator>(const mlx::core::array &a, T b)
    // {
    //     return greater(a, array(b));
    // }

    /** Returns bool std::unique_ptr<mlx::core::array> with (a >= b) element-wise. */
    std::unique_ptr<mlx::core::array> greater_equal(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // inline std::unique_ptr<mlx::core::array> operator>=(const mlx::core::array &a, const mlx::core::array &b)
    // {
    //     return mlx_cxx::greater_equal(a, b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator>=(T a, const mlx::core::array &b)
    // {
    //     return greater_equal(array(a), b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator>=(const mlx::core::array &a, T b)
    // {
    //     return greater_equal(a, array(b));
    // }

    /** Returns bool std::unique_ptr<mlx::core::array> with (a < b) element-wise. */
    std::unique_ptr<mlx::core::array> less(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // inline std::unique_ptr<mlx::core::array> operator<(const mlx::core::array &a, const mlx::core::array &b)
    // {
    //     return mlx_cxx::less(a, b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator<(T a, const mlx::core::array &b)
    // {
    //     return less(array(a), b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator<(const mlx::core::array &a, T b)
    // {
    //     return less(a, array(b));
    // }

    /** Returns bool std::unique_ptr<mlx::core::array> with (a <= b) element-wise. */
    std::unique_ptr<mlx::core::array> less_equal(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // inline std::unique_ptr<mlx::core::array> operator<=(const mlx::core::array &a, const mlx::core::array &b)
    // {
    //     return mlx_cxx::less_equal(a, b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator<=(T a, const mlx::core::array &b)
    // {
    //     return less_equal(array(a), b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator<=(const mlx::core::array &a, T b)
    // {
    //     return less_equal(a, array(b));
    // }

    /** True if two arrays have the same shape and elements. */
    std::unique_ptr<mlx::core::array> array_equal(
        const mlx::core::array &a,
        const mlx::core::array &b,
        bool equal_nan,
        mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array>
    array_equal(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {})
    {
        return array_equal(a, b, false, s);
    }

    std::unique_ptr<mlx::core::array> isnan(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> isinf(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> isposinf(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> isneginf(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Select from x or y depending on condition. */
    std::unique_ptr<mlx::core::array> where(
        const mlx::core::array &condition,
        const mlx::core::array &x,
        const mlx::core::array &y,
        mlx_cxx::StreamOrDevice s = {});

    /** Reduction operations */

    /** True if all elements in the std::unique_ptr<mlx::core::array> are true (or non-zero). **/
    std::unique_ptr<mlx::core::array> all(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> all(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::all(a, false, to_stream(s));
        return std::make_unique<mlx::core::array>(array);
    }

    /** True if the two arrays are equal within the specified tolerance. */
    std::unique_ptr<mlx::core::array> allclose(
        const mlx::core::array &a,
        const mlx::core::array &b,
        double rtol = 1e-5,
        double atol = 1e-8,
        bool equal_nan = false,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> isclose(
        const mlx::core::array& a,
        const mlx::core::array& b,
        double rtol = 1e-5,
        double atol = 1e-8,
        bool equal_nan = false,
        mlx_cxx::StreamOrDevice s = {});

    /**
     *  Reduces the input along the given axes. An output value is true
     *  if all the corresponding inputs are true.
     **/
    std::unique_ptr<mlx::core::array> all(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /**
     *  Reduces the input along the given axis. An output value is true
     *  if all the corresponding inputs are true.
     **/
    std::unique_ptr<mlx::core::array> all(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** True if any elements in the std::unique_ptr<mlx::core::array> are true (or non-zero). **/
    std::unique_ptr<mlx::core::array> any(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> any(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        auto array = any(a, false, to_stream(s));
        return std::make_unique<mlx::core::array>(array);
    }

    /**
     *  Reduces the input along the given axes. An output value is true
     *  if any of the corresponding inputs are true.
     **/
    std::unique_ptr<mlx::core::array> any(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /**
     *  Reduces the input along the given axis. An output value is true
     *  if any of the corresponding inputs are true.
     **/
    std::unique_ptr<mlx::core::array> any(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** Sums the elements of an array. */
    std::unique_ptr<mlx::core::array> sum(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> sum(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::sum(a, false, to_stream(s));
        return std::make_unique<mlx::core::array>(array);
    }

    /** Sums the elements of an std::unique_ptr<mlx::core::array> along the given axes. */
    std::unique_ptr<mlx::core::array> sum(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** Sums the elements of an std::unique_ptr<mlx::core::array> along the given axis. */
    std::unique_ptr<mlx::core::array> sum(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** Computes the mean of the elements of an array. */
    std::unique_ptr<mlx::core::array> mean(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> mean(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mean(a, false, to_stream(s));
        return std::make_unique<mlx::core::array>(array);
    }

    /** Computes the mean of the elements of an std::unique_ptr<mlx::core::array> along the given axes */
    std::unique_ptr<mlx::core::array> mean(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** Computes the mean of the elements of an std::unique_ptr<mlx::core::array> along the given axis */
    std::unique_ptr<mlx::core::array> mean(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** Computes the mean of the elements of an array. */
    std::unique_ptr<mlx::core::array> var(const mlx::core::array &a, bool keepdims, int ddof = 0, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> var(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        auto array = var(a, false, 0, to_stream(s));
        return std::make_unique<mlx::core::array>(array);
    }

    /** Computes the var of the elements of an std::unique_ptr<mlx::core::array> along the given axes */
    std::unique_ptr<mlx::core::array> var(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims = false,
        int ddof = 0,
        mlx_cxx::StreamOrDevice s = {});

    /** Computes the var of the elements of an std::unique_ptr<mlx::core::array> along the given axis */
    std::unique_ptr<mlx::core::array> var(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        int ddof = 0,
        mlx_cxx::StreamOrDevice s = {});

    /** The product of all elements of the array. */
    std::unique_ptr<mlx::core::array> prod(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> prod(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        auto array = prod(a, false, to_stream(s));
        return std::make_unique<mlx::core::array>(array);
    }

    /** The product of the elements of an std::unique_ptr<mlx::core::array> along the given axes. */
    std::unique_ptr<mlx::core::array> prod(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** The product of the elements of an std::unique_ptr<mlx::core::array> along the given axis. */
    std::unique_ptr<mlx::core::array> prod(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** The maximum of all elements of the array. */
    std::unique_ptr<mlx::core::array> max(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> max(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        auto array = max(a, false, to_stream(s));
        return std::make_unique<mlx::core::array>(array);
    }

    /** The maximum of the elements of an std::unique_ptr<mlx::core::array> along the given axes. */
    std::unique_ptr<mlx::core::array> max(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** The maximum of the elements of an std::unique_ptr<mlx::core::array> along the given axis. */
    std::unique_ptr<mlx::core::array> max(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** The minimum of all elements of the array. */
    std::unique_ptr<mlx::core::array> min(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> min(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        auto array = min(a, false, to_stream(s));
        return std::make_unique<mlx::core::array>(array);
    }

    /** The minimum of the elements of an std::unique_ptr<mlx::core::array> along the given axes. */
    std::unique_ptr<mlx::core::array> min(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** The minimum of the elements of an std::unique_ptr<mlx::core::array> along the given axis. */
    std::unique_ptr<mlx::core::array> min(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** Returns the index of the minimum value in the array. */
    std::unique_ptr<mlx::core::array> argmin(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> argmin(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        return argmin(a, false, s);
    }

    /** Returns the indices of the minimum values along a given axis. */
    std::unique_ptr<mlx::core::array> argmin(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** Returns the index of the maximum value in the array. */
    std::unique_ptr<mlx::core::array> argmax(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> argmax(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        return argmax(a, false, s);
    }

    /** Returns the indices of the maximum values along a given axis. */
    std::unique_ptr<mlx::core::array> argmax(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** Returns a sorted copy of the flattened array. */
    std::unique_ptr<mlx::core::array> sort(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Returns a sorted copy of the std::unique_ptr<mlx::core::array> along a given axis. */
    std::unique_ptr<mlx::core::array> sort(const mlx::core::array &a, int axis, mlx_cxx::StreamOrDevice s = {});

    /** Returns indices that sort the flattened array. */
    std::unique_ptr<mlx::core::array> argsort(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Returns indices that sort the std::unique_ptr<mlx::core::array> along a given axis. */
    std::unique_ptr<mlx::core::array> argsort(const mlx::core::array &a, int axis, mlx_cxx::StreamOrDevice s = {});

    /**
     * Returns a partitioned copy of the flattened array
     * such that the smaller kth elements are first.
     **/
    std::unique_ptr<mlx::core::array> partition(const mlx::core::array &a, int kth, mlx_cxx::StreamOrDevice s = {});

    /**
     * Returns a partitioned copy of the std::unique_ptr<mlx::core::array> along a given axis
     * such that the smaller kth elements are first.
     **/
    std::unique_ptr<mlx::core::array> partition(const mlx::core::array &a, int kth, int axis, mlx_cxx::StreamOrDevice s = {});

    /**
     * Returns indices that partition the flattened array
     * such that the smaller kth elements are first.
     **/
    std::unique_ptr<mlx::core::array> argpartition(const mlx::core::array &a, int kth, mlx_cxx::StreamOrDevice s = {});

    /**
     * Returns indices that partition the std::unique_ptr<mlx::core::array> along a given axis
     * such that the smaller kth elements are first.
     **/
    std::unique_ptr<mlx::core::array> argpartition(const mlx::core::array &a, int kth, int axis, mlx_cxx::StreamOrDevice s = {});

    /** Returns topk elements of the flattened array. */
    std::unique_ptr<mlx::core::array> topk(const mlx::core::array &a, int k, mlx_cxx::StreamOrDevice s = {});

    /** Returns topk elements of the std::unique_ptr<mlx::core::array> along a given axis. */
    std::unique_ptr<mlx::core::array> topk(const mlx::core::array &a, int k, int axis, mlx_cxx::StreamOrDevice s = {});

    /** The logsumexp of all elements of the array. */
    std::unique_ptr<mlx::core::array> logsumexp(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> logsumexp(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        auto array = logsumexp(a, false, to_stream(s));
        return std::make_unique<mlx::core::array>(array);
    }

    /** The logsumexp of the elements of an std::unique_ptr<mlx::core::array> along the given axes. */
    std::unique_ptr<mlx::core::array> logsumexp(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** The logsumexp of the elements of an std::unique_ptr<mlx::core::array> along the given axis. */
    std::unique_ptr<mlx::core::array> logsumexp(
        const mlx::core::array &a,
        int axis,
        bool keepdims = false,
        mlx_cxx::StreamOrDevice s = {});

    /** Simple arithmetic operations */

    /** Absolute value of elements in an array. */
    std::unique_ptr<mlx::core::array> abs(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Negate an array. */
    std::unique_ptr<mlx::core::array> negative(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});
    // std::unique_ptr<mlx::core::array> operator-(const mlx::core::array &a);

    /** The sign of the elements in an array. */
    std::unique_ptr<mlx::core::array> sign(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Logical not of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> logical_not(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Logical and of two arrays */
    std::unique_ptr<mlx::core::array> logical_and(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // std::unique_ptr<mlx::core::array> operator&&(const mlx::core::array &a, const mlx::core::array &b);

    /** Logical or of two arrays */
    std::unique_ptr<mlx::core::array> logical_or(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // std::unique_ptr<mlx::core::array> operator||(const mlx::core::array &a, const mlx::core::array &b);

    /** The reciprocal (1/x) of the elements in an array. */
    std::unique_ptr<mlx::core::array> reciprocal(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Add two arrays. */
    std::unique_ptr<mlx::core::array> add(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // std::unique_ptr<mlx::core::array> operator+(const mlx::core::array &a, const mlx::core::array &b);
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator+(T a, const mlx::core::array &b)
    // {
    //     return add(array(a), b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator+(const mlx::core::array &a, T b)
    // {
    //     return add(a, array(b));
    // }

    /** Subtract two arrays. */
    std::unique_ptr<mlx::core::array> subtract(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // std::unique_ptr<mlx::core::array> operator-(const mlx::core::array &a, const mlx::core::array &b);
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator-(T a, const mlx::core::array &b)
    // {
    //     return subtract(array(a), b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator-(const mlx::core::array &a, T b)
    // {
    //     return subtract(a, array(b));
    // }

    /** Multiply two arrays. */
    std::unique_ptr<mlx::core::array> multiply(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // std::unique_ptr<mlx::core::array> operator*(const mlx::core::array &a, const mlx::core::array &b);
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator*(T a, const mlx::core::array &b)
    // {
    //     return multiply(array(a), b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator*(const mlx::core::array &a, T b)
    // {
    //     return multiply(a, array(b));
    // }

    /** Divide two arrays. */
    std::unique_ptr<mlx::core::array> divide(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // std::unique_ptr<mlx::core::array> operator/(const mlx::core::array &a, const mlx::core::array &b);
    // std::unique_ptr<mlx::core::array> operator/(double a, const mlx::core::array &b);
    // std::unique_ptr<mlx::core::array> operator/(const mlx::core::array &a, double b);

    /** Compute the element-wise quotient and remainder. */
    std::unique_ptr<std::vector<mlx::core::array>>
    divmod(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});

    /** Compute integer division. Equivalent to doing floor(a / x). */
    std::unique_ptr<mlx::core::array> floor_divide(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});

    /** Compute the element-wise remainder of division */
    std::unique_ptr<mlx::core::array> remainder(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // std::unique_ptr<mlx::core::array> operator%(const mlx::core::array &a, const mlx::core::array &b);
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator%(T a, const mlx::core::array &b)
    // {
    //     return remainder(array(a), b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator%(const mlx::core::array &a, T b)
    // {
    //     return remainder(a, array(b));
    // }

    /** Element-wise maximum between two arrays. */
    std::unique_ptr<mlx::core::array> maximum(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});

    /** Element-wise minimum between two arrays. */
    std::unique_ptr<mlx::core::array> minimum(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});

    /** Floor the element of an array. **/
    std::unique_ptr<mlx::core::array> floor(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Ceil the element of an array. **/
    std::unique_ptr<mlx::core::array> ceil(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Square the elements of an array. */
    std::unique_ptr<mlx::core::array> square(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Exponential of the elements of an array. */
    std::unique_ptr<mlx::core::array> exp(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Sine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> sin(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Cosine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> cos(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Tangent of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> tan(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Arc Sine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arcsin(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Arc Cosine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arccos(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Arc Tangent of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arctan(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Hyperbolic Sine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> sinh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Hyperbolic Cosine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> cosh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Hyperbolic Tangent of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> tanh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Inverse Hyperbolic Sine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arcsinh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Inverse Hyperbolic Cosine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arccosh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Inverse Hyperbolic Tangent of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arctanh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Natural logarithm of the elements of an array. */
    std::unique_ptr<mlx::core::array> log(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Log base 2 of the elements of an array. */
    std::unique_ptr<mlx::core::array> log2(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Log base 10 of the elements of an array. */
    std::unique_ptr<mlx::core::array> log10(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Natural logarithm of one plus elements in the array: `log(1 + a)`. */
    std::unique_ptr<mlx::core::array> log1p(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Log-add-exp of one elements in the array: `log(exp(a) + exp(b))`. */
    std::unique_ptr<mlx::core::array> logaddexp(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});

    /** Element-wise logistic sigmoid of the array: `1 / (1 + exp(-x)`. */
    std::unique_ptr<mlx::core::array> sigmoid(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Computes the error function of the elements of an array. */
    std::unique_ptr<mlx::core::array> erf(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Computes the inverse error function of the elements of an array. */
    std::unique_ptr<mlx::core::array> erfinv(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Stop the flow of gradients. */
    std::unique_ptr<mlx::core::array> stop_gradient(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Round a floating point number */
    std::unique_ptr<mlx::core::array> round(const mlx::core::array &a, int decimals, mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> round(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {})
    {
        return round(a, 0, s);
    }

    /** Matrix-matrix multiplication. */
    std::unique_ptr<mlx::core::array> matmul(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});

    /** Gather std::unique_ptr<mlx::core::array> entries given indices and slices */
    std::unique_ptr<mlx::core::array> gather(
        const mlx::core::array &a,
        const std::vector<mlx::core::array>& indices,
        const std::vector<int> &axes,
        const std::vector<int> &slice_sizes,
        mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> gather(
        const mlx::core::array &a,
        const mlx::core::array &indices,
        int axis,
        const std::vector<int> &slice_sizes,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::gather(a, indices, axis, slice_sizes, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Take std::unique_ptr<mlx::core::array> slices at the given indices of the specified axis. */
    std::unique_ptr<mlx::core::array> take(
        const mlx::core::array &a,
        const mlx::core::array &indices,
        int axis,
        mlx_cxx::StreamOrDevice s = {});

    /** Take std::unique_ptr<mlx::core::array> entries at the given indices treating the std::unique_ptr<mlx::core::array> as flattened. */
    std::unique_ptr<mlx::core::array> take(const mlx::core::array &a, const mlx::core::array &indices, mlx_cxx::StreamOrDevice s = {});

    /** Take std::unique_ptr<mlx::core::array> entries given indices along the axis */
    std::unique_ptr<mlx::core::array> take_along_axis(
        const mlx::core::array &a,
        const mlx::core::array &indices,
        int axis,
        mlx_cxx::StreamOrDevice s = {});

    /** Scatter updates to given linear indices */
    std::unique_ptr<mlx::core::array> scatter(
        const mlx::core::array &a,
        const std::vector<mlx::core::array>& indices,
        const mlx::core::array &updates,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> scatter(
        const mlx::core::array &a,
        const mlx::core::array &indices,
        const mlx::core::array &updates,
        int axis,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::scatter(a, indices, updates, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Scatter and add updates to given indices */
    std::unique_ptr<mlx::core::array> scatter_add(
        const mlx::core::array &a,
        const std::vector<mlx::core::array>& indices,
        const mlx::core::array &updates,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> scatter_add(
        const mlx::core::array &a,
        const mlx::core::array &indices,
        const mlx::core::array &updates,
        int axis,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::scatter_add(a, indices, updates, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Scatter and prod updates to given indices */
    std::unique_ptr<mlx::core::array> scatter_prod(
        const mlx::core::array &a,
        const std::vector<mlx::core::array>& indices,
        const mlx::core::array &updates,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> scatter_prod(
        const mlx::core::array &a,
        const mlx::core::array &indices,
        const mlx::core::array &updates,
        int axis,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::scatter_prod(a, indices, updates, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Scatter and max updates to given linear indices */
    std::unique_ptr<mlx::core::array> scatter_max(
        const mlx::core::array &a,
        const std::vector<mlx::core::array>& indices,
        const mlx::core::array &updates,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> scatter_max(
        const mlx::core::array &a,
        const mlx::core::array &indices,
        const mlx::core::array &updates,
        int axis,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::scatter_max(a, indices, updates, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    /** Scatter and min updates to given linear indices */
    std::unique_ptr<mlx::core::array> scatter_min(
        const mlx::core::array &a,
        const std::vector<mlx::core::array>& indices,
        const mlx::core::array &updates,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});
    inline std::unique_ptr<mlx::core::array> scatter_min(
        const mlx::core::array &a,
        const mlx::core::array &indices,
        const mlx::core::array &updates,
        int axis,
        mlx_cxx::StreamOrDevice s = {})
    {
        auto array = mlx::core::scatter_min(a, indices, updates, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Square root the elements of an array. */
    std::unique_ptr<mlx::core::array> sqrt(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Square root and reciprocal the elements of an array. */
    std::unique_ptr<mlx::core::array> rsqrt(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Softmax of an array. */
    std::unique_ptr<mlx::core::array> softmax(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s = {});

    /** Softmax of an array. */
    std::unique_ptr<mlx::core::array> softmax(const mlx::core::array &a, mlx_cxx::StreamOrDevice s = {});

    /** Softmax of an array. */
    inline std::unique_ptr<mlx::core::array> softmax(const mlx::core::array &a, int axis, mlx_cxx::StreamOrDevice s = {})
    {
        return softmax(a, std::vector<int>{axis}, s);
    }

    /** Raise elements of a to the power of b element-wise */
    std::unique_ptr<mlx::core::array> power(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});
    // inline std::unique_ptr<mlx::core::array> operator^(const mlx::core::array &a, const mlx::core::array &b)
    // {
    //     return mlx_cxx::power(a, b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator^(T a, const mlx::core::array &b)
    // {
    //     return power(array(a), b);
    // }
    // template <typename T>
    // std::unique_ptr<mlx::core::array> operator^(const mlx::core::array &a, T b)
    // {
    //     return power(a, array(b));
    // }

    /** Cumulative sum of an array. */
    std::unique_ptr<mlx::core::array> cumsum(
        const mlx::core::array &a,
        int axis,
        bool reverse = false,
        bool inclusive = true,
        mlx_cxx::StreamOrDevice s = {});

    /** Cumulative product of an array. */
    std::unique_ptr<mlx::core::array> cumprod(
        const mlx::core::array &a,
        int axis,
        bool reverse = false,
        bool inclusive = true,
        mlx_cxx::StreamOrDevice s = {});

    /** Cumulative max of an array. */
    std::unique_ptr<mlx::core::array> cummax(
        const mlx::core::array &a,
        int axis,
        bool reverse = false,
        bool inclusive = true,
        mlx_cxx::StreamOrDevice s = {});

    /** Cumulative min of an array. */
    std::unique_ptr<mlx::core::array> cummin(
        const mlx::core::array &a,
        int axis,
        bool reverse = false,
        bool inclusive = true,
        mlx_cxx::StreamOrDevice s = {});

    /** Convolution operations */

    /** 1D convolution with a filter */
    std::unique_ptr<mlx::core::array> conv1d(
        const mlx::core::array &input,
        const mlx::core::array &weight,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        int groups = 1,
        mlx_cxx::StreamOrDevice s = {});

    /** 2D convolution with a filter */
    std::unique_ptr<mlx::core::array> conv2d(
        const mlx::core::array &input,
        const mlx::core::array &weight,
        const std::array<int, 2> &stride = {1, 1},
        const std::array<int, 2> &padding = {0, 0},
        const std::array<int, 2> &dilation = {1, 1},
        int groups = 1,
        mlx_cxx::StreamOrDevice s = {});

    /** Quantized matmul multiplies x with a quantized matrix w*/
    std::unique_ptr<mlx::core::array> quantized_matmul(
        const mlx::core::array &x,
        const mlx::core::array &w,
        const mlx::core::array &scales,
        const mlx::core::array &biases,
        bool transpose = true,
        int group_size = 64,
        int bits = 4,
        mlx_cxx::StreamOrDevice s = {});

    /** Quantize a matrix along its last axis */
    std::array<std::unique_ptr<mlx::core::array>, 3> quantize(
        const mlx::core::array &w,
        int group_size = 64,
        int bits = 4,
        mlx_cxx::StreamOrDevice s = {});

    /** Dequantize a matrix produced by quantize() */
    std::unique_ptr<mlx::core::array> dequantize(
        const mlx::core::array &w,
        const mlx::core::array &scales,
        const mlx::core::array &biases,
        int group_size = 64,
        int bits = 4,
        mlx_cxx::StreamOrDevice s = {});

    /** TensorDot returns a contraction of a and b over multiple dimensions. */
    std::unique_ptr<mlx::core::array> tensordot(
        const mlx::core::array &a,
        const mlx::core::array &b,
        const int dims = 2,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> tensordot(
        const mlx::core::array &a,
        const mlx::core::array &b,
        const std::array<std::unique_ptr<std::vector<int>>, 2> &dims,
        mlx_cxx::StreamOrDevice s = {});

    /** Compute the outer product of two vectors. */
    std::unique_ptr<mlx::core::array> outer(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});

    /** Compute the inner product of two vectors. */
    std::unique_ptr<mlx::core::array> inner(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s = {});

    /** Compute D = beta * C + alpha * (A @ B) */
    std::unique_ptr<mlx::core::array> addmm(
        std::unique_ptr<mlx::core::array> c,
        std::unique_ptr<mlx::core::array> a,
        std::unique_ptr<mlx::core::array> b,
        const float& alpha = 1.f,
        const float& beta = 1.f,
        mlx_cxx::StreamOrDevice s = {});

    /** Extract a diagonal or construct a diagonal array */
    std::unique_ptr<mlx::core::array> diagonal(
        const mlx::core::array& a,
        int offset = 0,
        int axis1 = 0,
        int axis2 = 1,
        mlx_cxx::StreamOrDevice s = {});

    /** Extract diagonal from a 2d array or create a diagonal matrix. */
    std::unique_ptr<mlx::core::array> diag(const mlx::core::array& a, int k = 0, mlx_cxx::StreamOrDevice s = {});

    /**
     * Implements the identity function but allows injecting dependencies to other
     * arrays. This ensures that these other arrays will have been computed
     * when the outputs of this function are computed.
     */
    std::unique_ptr<std::vector<mlx::core::array>> depends(
        const std::vector<mlx::core::array>& inputs,
        const std::vector<mlx::core::array>& dependencies);
}