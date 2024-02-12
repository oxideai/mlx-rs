#include "mlx/ops.h"

#include "mlx-cxx/ops.hpp"

namespace mlx_cxx
{
    std::optional<mlx::core::array> to_std_optional(const OptionalArray &opt)
    {
        switch (opt.tag)
        {
        case OptionalArray::Tag::None:
            return std::nullopt;
        case OptionalArray::Tag::Some:
            return *opt.payload.some;
        }
    }

    /**
     * A 1D std::unique_ptr<mlx::core::array> of numbers starting at `start` (optional),
     * stopping at stop, stepping by `step` (optional). */
    std::unique_ptr<mlx::core::array> arange(
        double start,
        double stop,
        double step,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arange(start, stop, step, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> arange(double start, double stop, double step, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arange(start, stop, step, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> arange(double start, double stop, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arange(start, stop, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> arange(double start, double stop, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arange(start, stop, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> arange(double stop, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arange(stop, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> arange(double stop, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arange(stop, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> arange(int start, int stop, int step, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arange(start, stop, step, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> arange(int start, int stop, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arange(start, stop, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> arange(int stop, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arange(stop, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** A 1D std::unique_ptr<mlx::core::array> of `num` evenly spaced numbers in the range `[start, stop]` */
    std::unique_ptr<mlx::core::array> linspace(
        double start,
        double stop,
        int num,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::linspace(start, stop, num, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Convert an std::unique_ptr<mlx::core::array> to the given data type. */
    std::unique_ptr<mlx::core::array> astype(const mlx::core::array &a, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::astype(a, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Create a view of an std::unique_ptr<mlx::core::array> with the given shape and strides. */
    std::unique_ptr<mlx::core::array> as_strided(
        const mlx::core::array &a,
        std::unique_ptr<std::vector<int>> shape,
        std::unique_ptr<std::vector<size_t>> strides,
        size_t offset,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::as_strided(a, *shape, *strides, offset, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Copy another array. */
    std::unique_ptr<mlx::core::array> copy(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::copy(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Fill an std::unique_ptr<mlx::core::array> of the given shape with the given value(s). */
    std::unique_ptr<mlx::core::array> full(
        const std::vector<int> &shape,
        const mlx::core::array &vals,
        mlx::core::Dtype dtype,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::full(shape, vals, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> full(
        const std::vector<int> &shape,
        const mlx::core::array &vals,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::full(shape, vals, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Fill an std::unique_ptr<mlx::core::array> of the given shape with zeros. */
    std::unique_ptr<mlx::core::array> zeros(const std::vector<int> &shape, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::zeros(shape, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> zeros_like(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::zeros_like(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Fill an std::unique_ptr<mlx::core::array> of the given shape with ones. */
    std::unique_ptr<mlx::core::array> ones(const std::vector<int> &shape, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::ones(shape, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> ones_like(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::ones_like(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Fill an std::unique_ptr<mlx::core::array> of the given shape (n,m) with ones in the specified diagonal
     * k, and zeros everywhere else. */
    std::unique_ptr<mlx::core::array> eye(int n, int m, int k, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::eye(n, m, k, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Create a square matrix of shape (n,n) of zeros, and ones in the major
     * diagonal. */
    std::unique_ptr<mlx::core::array> identity(int n, mlx::core::Dtype dtype, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::identity(n, dtype, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> tri(int n, int m, int k, mlx::core::Dtype type, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::tri(n, m, k, type, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> tril(std::unique_ptr<mlx::core::array> x, int k, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::tril(*x, k, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> triu(std::unique_ptr<mlx::core::array> x, int k, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::triu(*x, k, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** std::unique_ptr<mlx::core::array> manipulation */

    /** Reshape an std::unique_ptr<mlx::core::array> to the given shape. */
    std::unique_ptr<mlx::core::array> reshape(const mlx::core::array &a, std::unique_ptr<std::vector<int>> shape, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::reshape(a, *shape, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Flatten the dimensions in the range `[start_axis, end_axis]` . */
    std::unique_ptr<mlx::core::array> flatten(
        const mlx::core::array &a,
        int start_axis,
        int end_axis,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::flatten(a, start_axis, end_axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Flatten the std::unique_ptr<mlx::core::array> to 1D. */
    std::unique_ptr<mlx::core::array> flatten(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::flatten(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Remove singleton dimensions at the given axes. */
    std::unique_ptr<mlx::core::array> squeeze(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::squeeze(a, axes, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Remove all singleton dimensions. */
    std::unique_ptr<mlx::core::array> squeeze(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::squeeze(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Add a singleton dimension at the given axes. */
    std::unique_ptr<mlx::core::array> expand_dims(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::expand_dims(a, axes, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Slice an array. */
    std::unique_ptr<mlx::core::array> slice(
        const mlx::core::array &a,
        std::unique_ptr<std::vector<int>> start,
        std::unique_ptr<std::vector<int>> stop,
        std::unique_ptr<std::vector<int>> strides,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::slice(a, *start, *stop, *strides, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Slice an std::unique_ptr<mlx::core::array> with a stride of 1 in each dimension. */
    std::unique_ptr<mlx::core::array> slice(
        const mlx::core::array &a,
        const std::vector<int> &start,
        const std::vector<int> &stop,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::slice(a, start, stop, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Split an std::unique_ptr<mlx::core::array> into sub-arrays along a given axis. */
    std::unique_ptr<std::vector<mlx::core::array>>
    split(const mlx::core::array &a, int num_splits, int axis, mlx_cxx::StreamOrDevice s)
    {
        auto arrays = mlx::core::split(a, num_splits, axis, s.to_variant());
        return std::make_unique<std::vector<mlx::core::array>>(arrays);
    }
    std::unique_ptr<std::vector<mlx::core::array>> split(const mlx::core::array &a, int num_splits, mlx_cxx::StreamOrDevice s)
    {
        auto arrays = mlx::core::split(a, num_splits, s.to_variant());
        return std::make_unique<std::vector<mlx::core::array>>(arrays);
    }
    std::unique_ptr<std::vector<mlx::core::array>> split(
        const mlx::core::array &a,
        const std::vector<int> &indices,
        int axis,
        mlx_cxx::StreamOrDevice s)
    {
        auto arrays = mlx::core::split(a, indices, axis, s.to_variant());
        return std::make_unique<std::vector<mlx::core::array>>(arrays);
    }
    std::unique_ptr<std::vector<mlx::core::array>>
    split(const mlx::core::array &a, const std::vector<int> &indices, mlx_cxx::StreamOrDevice s)
    {
        auto arrays = mlx::core::split(a, indices, s.to_variant());
        return std::make_unique<std::vector<mlx::core::array>>(arrays);
    }

    /**
     * Clip (limit) the values in an array.
     */
    std::unique_ptr<mlx::core::array> clip(
        const mlx::core::array &a,
        const OptionalArray &a_min,
        const OptionalArray &a_max,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::clip(a, to_std_optional(a_min), to_std_optional(a_max), s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Concatenate arrays along a given axis. */
    std::unique_ptr<mlx::core::array> concatenate(
        rust::Slice<const std::unique_ptr<mlx::core::array>> arrays,
        int axis,
        mlx_cxx::StreamOrDevice s)
    {
        std::vector<mlx::core::array> copy_constructed_arrays;
        for (auto &array : arrays)
        {
            copy_constructed_arrays.push_back(*array);
        }
        auto array = mlx::core::concatenate(copy_constructed_arrays, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> concatenate(rust::Slice<const std::unique_ptr<mlx::core::array>> arrays, mlx_cxx::StreamOrDevice s)
    {
        auto copy_constructed_arrays = std::vector<mlx::core::array>();
        for (auto &array : arrays)
        {
            copy_constructed_arrays.push_back(*array);
        }
        auto array = mlx::core::concatenate(copy_constructed_arrays, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Stack arrays along a new axis. */
    std::unique_ptr<mlx::core::array> stack(rust::Slice<const std::unique_ptr<mlx::core::array>> arrays, int axis, mlx_cxx::StreamOrDevice s)
    {
        auto copy_constructed_arrays = std::vector<mlx::core::array>();
        for (auto &array : arrays)
        {
            copy_constructed_arrays.push_back(*array);
        }
        auto array = mlx::core::stack(copy_constructed_arrays, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> stack(rust::Slice<const std::unique_ptr<mlx::core::array>> arrays, mlx_cxx::StreamOrDevice s)
    {
        auto copy_constructed_arrays = std::vector<mlx::core::array>();
        for (auto &array : arrays)
        {
            copy_constructed_arrays.push_back(*array);
        }
        auto array = mlx::core::stack(copy_constructed_arrays, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Repeat an std::unique_ptr<mlx::core::array> along an axis. */
    std::unique_ptr<mlx::core::array> repeat(const mlx::core::array &arr, int repeats, int axis, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::repeat(arr, repeats, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> repeat(const mlx::core::array &arr, int repeats, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::repeat(arr, repeats, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> tile(
        const mlx::core::array &arr,
        std::unique_ptr<std::vector<int>> reps,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::tile(arr, *reps, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Permutes the dimensions according to the given axes. */
    std::unique_ptr<mlx::core::array> transpose(const mlx::core::array &a, std::unique_ptr<std::vector<int>> axes, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::transpose(a, *axes, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Swap two axes of an array. */
    std::unique_ptr<mlx::core::array> swapaxes(const mlx::core::array &a, int axis1, int axis2, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::swapaxes(a, axis1, axis2, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Move an axis of an array. */
    std::unique_ptr<mlx::core::array> moveaxis(
        const mlx::core::array &a,
        int source,
        int destination,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::moveaxis(a, source, destination, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Pad an std::unique_ptr<mlx::core::array> with a constant value */
    std::unique_ptr<mlx::core::array> pad(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        const std::vector<int> &low_pad_size,
        const std::vector<int> &high_pad_size,
        const mlx::core::array &pad_value,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::pad(a, axes, low_pad_size, high_pad_size, pad_value, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Pad an std::unique_ptr<mlx::core::array> with a constant value along all axes */
    std::unique_ptr<mlx::core::array> pad(
        const mlx::core::array &a,
        rust::Slice<const std::array<int, 2>> pad_width,
        const mlx::core::array &pad_value,
        mlx_cxx::StreamOrDevice s)
    {
        // convert std::vector<std::array<int, 2>> to std::vector<std::pair<int, int>>
        auto pad_width_pair = std::vector<std::pair<int, int>>();
        for (auto &pair : pad_width)
        {
            pad_width_pair.push_back({pair[0], pair[1]});
        }
        auto array = mlx::core::pad(a, pad_width_pair, pad_value, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> pad(
        const mlx::core::array &a,
        const std::array<int, 2> &pad_width,
        const mlx::core::array &pad_value,
        mlx_cxx::StreamOrDevice s)
    {
        // Convert std::array<int, 2> to std::pair<int, int>
        auto pad_width_pair = std::pair<int, int>(pad_width[0], pad_width[1]);
        auto array = mlx::core::pad(a, pad_width_pair, pad_value, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }
    std::unique_ptr<mlx::core::array> pad(
        const mlx::core::array &a,
        int pad_width,
        const mlx::core::array &pad_value,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::pad(a, pad_width, pad_value, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Permutes the dimensions in reverse order. */
    std::unique_ptr<mlx::core::array> transpose(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::transpose(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Broadcast an std::unique_ptr<mlx::core::array> to a given shape. */
    std::unique_ptr<mlx::core::array> broadcast_to(
        const mlx::core::array &a,
        const std::vector<int> &shape,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::broadcast_to(a, shape, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Broadcast a vector of arrays against one another. */
    std::unique_ptr<std::vector<mlx::core::array>> broadcast_arrays(
        rust::Slice<const std::unique_ptr<mlx::core::array>> inputs,
        mlx_cxx::StreamOrDevice s)
    {
        auto copy_constructed_inputs = std::vector<mlx::core::array>();
        for (auto &input : inputs)
        {
            copy_constructed_inputs.push_back(*input);
        }
        auto arrays = mlx::core::broadcast_arrays(copy_constructed_inputs, s.to_variant());
        return std::make_unique<std::vector<mlx::core::array>>(arrays);
    }

    /** Comparison operations */

    /** Returns the bool std::unique_ptr<mlx::core::array> with (a == b) element-wise. */
    std::unique_ptr<mlx::core::array> equal(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::equal(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns the bool std::unique_ptr<mlx::core::array> with (a != b) element-wise. */
    std::unique_ptr<mlx::core::array> not_equal(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::not_equal(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns bool std::unique_ptr<mlx::core::array> with (a > b) element-wise. */
    std::unique_ptr<mlx::core::array> greater(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::greater(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns bool std::unique_ptr<mlx::core::array> with (a >= b) element-wise. */
    std::unique_ptr<mlx::core::array> greater_equal(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::greater_equal(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns bool std::unique_ptr<mlx::core::array> with (a < b) element-wise. */
    std::unique_ptr<mlx::core::array> less(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::less(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns bool std::unique_ptr<mlx::core::array> with (a <= b) element-wise. */
    std::unique_ptr<mlx::core::array> less_equal(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::less_equal(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** True if two arrays have the same shape and elements. */
    std::unique_ptr<mlx::core::array> array_equal(
        const mlx::core::array &a,
        const mlx::core::array &b,
        bool equal_nan,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::array_equal(a, b, equal_nan, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> isnan(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::isnan(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> isinf(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::isinf(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> isposinf(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::isposinf(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> isneginf(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::isneginf(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Select from x or y depending on condition. */
    std::unique_ptr<mlx::core::array> where(
        const mlx::core::array &condition,
        const mlx::core::array &x,
        const mlx::core::array &y,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::where(condition, x, y, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Reduction operations */

    /** True if all elements in the std::unique_ptr<mlx::core::array> are true (or non-zero). **/
    std::unique_ptr<mlx::core::array> all(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::all(a, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** True if the two arrays are equal within the specified tolerance. */
    std::unique_ptr<mlx::core::array> allclose(
        const mlx::core::array &a,
        const mlx::core::array &b,
        double rtol,
        double atol,
        bool equal_nan,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::allclose(a, b, rtol, atol, equal_nan, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> isclose(
        const mlx::core::array &a,
        const mlx::core::array &b,
        double rtol,
        double atol,
        bool equal_nan,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::isclose(a, b, rtol, atol, equal_nan, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /**
     *  Reduces the input along the given axes. An output value is true
     *  if all the corresponding inputs are true.
     **/
    std::unique_ptr<mlx::core::array> all(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::all(a, axes, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /**
     *  Reduces the input along the given axis. An output value is true
     *  if all the corresponding inputs are true.
     **/
    std::unique_ptr<mlx::core::array> all(
        const mlx::core::array &a,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::all(a, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** True if any elements in the std::unique_ptr<mlx::core::array> are true (or non-zero). **/
    std::unique_ptr<mlx::core::array> any(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::any(a, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /**
     *  Reduces the input along the given axes. An output value is true
     *  if any of the corresponding inputs are true.
     **/
    std::unique_ptr<mlx::core::array> any(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::any(a, axes, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /**
     *  Reduces the input along the given axis. An output value is true
     *  if any of the corresponding inputs are true.
     **/
    std::unique_ptr<mlx::core::array> any(
        const mlx::core::array &a,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::any(a, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Sums the elements of an array. */
    std::unique_ptr<mlx::core::array> sum(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::sum(a, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Sums the elements of an std::unique_ptr<mlx::core::array> along the given axes. */
    std::unique_ptr<mlx::core::array> sum(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::sum(a, axes, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Sums the elements of an std::unique_ptr<mlx::core::array> along the given axis. */
    std::unique_ptr<mlx::core::array> sum(
        const mlx::core::array &a,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::sum(a, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Computes the mean of the elements of an array. */
    std::unique_ptr<mlx::core::array> mean(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::mean(a, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Computes the mean of the elements of an std::unique_ptr<mlx::core::array> along the given axes */
    std::unique_ptr<mlx::core::array> mean(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::mean(a, axes, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Computes the mean of the elements of an std::unique_ptr<mlx::core::array> along the given axis */
    std::unique_ptr<mlx::core::array> mean(
        const mlx::core::array &a,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::mean(a, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Computes the mean of the elements of an array. */
    std::unique_ptr<mlx::core::array> var(const mlx::core::array &a, bool keepdims, int ddof, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::var(a, keepdims, ddof, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Computes the var of the elements of an std::unique_ptr<mlx::core::array> along the given axes */
    std::unique_ptr<mlx::core::array> var(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims,
        int ddof,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::var(a, axes, keepdims, ddof, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Computes the var of the elements of an std::unique_ptr<mlx::core::array> along the given axis */
    std::unique_ptr<mlx::core::array> var(
        const mlx::core::array &a,
        int axis,
        bool keepdims,
        int ddof,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::var(a, axis, keepdims, ddof, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The product of all elements of the array. */
    std::unique_ptr<mlx::core::array> prod(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::prod(a, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The product of the elements of an std::unique_ptr<mlx::core::array> along the given axes. */
    std::unique_ptr<mlx::core::array> prod(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::prod(a, axes, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The product of the elements of an std::unique_ptr<mlx::core::array> along the given axis. */
    std::unique_ptr<mlx::core::array> prod(
        const mlx::core::array &a,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::prod(a, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The maximum of all elements of the array. */
    std::unique_ptr<mlx::core::array> max(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::max(a, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The maximum of the elements of an std::unique_ptr<mlx::core::array> along the given axes. */
    std::unique_ptr<mlx::core::array> max(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::max(a, axes, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The maximum of the elements of an std::unique_ptr<mlx::core::array> along the given axis. */
    std::unique_ptr<mlx::core::array> max(
        const mlx::core::array &a,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::max(a, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The minimum of all elements of the array. */
    std::unique_ptr<mlx::core::array> min(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::min(a, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The minimum of the elements of an std::unique_ptr<mlx::core::array> along the given axes. */
    std::unique_ptr<mlx::core::array> min(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::min(a, axes, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The minimum of the elements of an std::unique_ptr<mlx::core::array> along the given axis. */
    std::unique_ptr<mlx::core::array> min(
        const mlx::core::array &a,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::min(a, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns the index of the minimum value in the array. */
    std::unique_ptr<mlx::core::array> argmin(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::argmin(a, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns the indices of the minimum values along a given axis. */
    std::unique_ptr<mlx::core::array> argmin(
        const mlx::core::array &a,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::argmin(a, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns the index of the maximum value in the array. */
    std::unique_ptr<mlx::core::array> argmax(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::argmax(a, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns the indices of the maximum values along a given axis. */
    std::unique_ptr<mlx::core::array> argmax(
        const mlx::core::array &a,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::argmax(a, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns a sorted copy of the flattened array. */
    std::unique_ptr<mlx::core::array> sort(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::sort(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns a sorted copy of the std::unique_ptr<mlx::core::array> along a given axis. */
    std::unique_ptr<mlx::core::array> sort(const mlx::core::array &a, int axis, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::sort(a, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns indices that sort the flattened array. */
    std::unique_ptr<mlx::core::array> argsort(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::argsort(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns indices that sort the std::unique_ptr<mlx::core::array> along a given axis. */
    std::unique_ptr<mlx::core::array> argsort(const mlx::core::array &a, int axis, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::argsort(a, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /**
     * Returns a partitioned copy of the flattened array
     * such that the smaller kth elements are first.
     **/
    std::unique_ptr<mlx::core::array> partition(const mlx::core::array &a, int kth, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::partition(a, kth, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /**
     * Returns a partitioned copy of the std::unique_ptr<mlx::core::array> along a given axis
     * such that the smaller kth elements are first.
     **/
    std::unique_ptr<mlx::core::array> partition(const mlx::core::array &a, int kth, int axis, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::partition(a, kth, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /**
     * Returns indices that partition the flattened array
     * such that the smaller kth elements are first.
     **/
    std::unique_ptr<mlx::core::array> argpartition(const mlx::core::array &a, int kth, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::argpartition(a, kth, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /**
     * Returns indices that partition the std::unique_ptr<mlx::core::array> along a given axis
     * such that the smaller kth elements are first.
     **/
    std::unique_ptr<mlx::core::array> argpartition(const mlx::core::array &a, int kth, int axis, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::argpartition(a, kth, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns topk elements of the flattened array. */
    std::unique_ptr<mlx::core::array> topk(const mlx::core::array &a, int k, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::topk(a, k, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Returns topk elements of the std::unique_ptr<mlx::core::array> along a given axis. */
    std::unique_ptr<mlx::core::array> topk(const mlx::core::array &a, int k, int axis, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::topk(a, k, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The logsumexp of all elements of the array. */
    std::unique_ptr<mlx::core::array> logsumexp(const mlx::core::array &a, bool keepdims, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::logsumexp(a, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The logsumexp of the elements of an std::unique_ptr<mlx::core::array> along the given axes. */
    std::unique_ptr<mlx::core::array> logsumexp(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::logsumexp(a, axes, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The logsumexp of the elements of an std::unique_ptr<mlx::core::array> along the given axis. */
    std::unique_ptr<mlx::core::array> logsumexp(
        const mlx::core::array &a,
        int axis,
        bool keepdims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::logsumexp(a, axis, keepdims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Simple arithmetic operations */

    /** Absolute value of elements in an array. */
    std::unique_ptr<mlx::core::array> abs(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::abs(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Negate an array. */
    std::unique_ptr<mlx::core::array> negative(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::negative(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The sign of the elements in an array. */
    std::unique_ptr<mlx::core::array> sign(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::sign(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Logical not of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> logical_not(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::logical_not(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Logical and of two arrays */
    std::unique_ptr<mlx::core::array> logical_and(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::logical_and(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Logical or of two arrays */
    std::unique_ptr<mlx::core::array> logical_or(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::logical_or(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** The reciprocal (1/x) of the elements in an array. */
    std::unique_ptr<mlx::core::array> reciprocal(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::reciprocal(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Add two arrays. */
    std::unique_ptr<mlx::core::array> add(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::add(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Subtract two arrays. */
    std::unique_ptr<mlx::core::array> subtract(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::subtract(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Multiply two arrays. */
    std::unique_ptr<mlx::core::array> multiply(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::multiply(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Divide two arrays. */
    std::unique_ptr<mlx::core::array> divide(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::divide(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Compute the element-wise quotient and remainder. */
    std::unique_ptr<std::vector<mlx::core::array>>
    divmod(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto arrays = mlx::core::divmod(a, b, s.to_variant());
        return std::make_unique<std::vector<mlx::core::array>>(arrays);
    }

    /** Compute integer division. Equivalent to doing floor(a / x). */
    std::unique_ptr<mlx::core::array> floor_divide(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::floor_divide(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Compute the element-wise remainder of division */
    std::unique_ptr<mlx::core::array> remainder(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::remainder(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Element-wise maximum between two arrays. */
    std::unique_ptr<mlx::core::array> maximum(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::maximum(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Element-wise minimum between two arrays. */
    std::unique_ptr<mlx::core::array> minimum(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::minimum(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Floor the element of an array. **/
    std::unique_ptr<mlx::core::array> floor(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::floor(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Ceil the element of an array. **/
    std::unique_ptr<mlx::core::array> ceil(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::ceil(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Square the elements of an array. */
    std::unique_ptr<mlx::core::array> square(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::square(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Exponential of the elements of an array. */
    std::unique_ptr<mlx::core::array> exp(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::exp(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Sine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> sin(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::sin(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Cosine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> cos(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::cos(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Tangent of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> tan(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::tan(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Arc Sine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arcsin(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arcsin(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Arc Cosine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arccos(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arccos(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Arc Tangent of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arctan(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arctan(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Hyperbolic Sine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> sinh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::sinh(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Hyperbolic Cosine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> cosh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::cosh(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Hyperbolic Tangent of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> tanh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::tanh(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Inverse Hyperbolic Sine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arcsinh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arcsinh(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Inverse Hyperbolic Cosine of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arccosh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arccosh(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Inverse Hyperbolic Tangent of the elements of an std::unique_ptr<mlx::core::array> */
    std::unique_ptr<mlx::core::array> arctanh(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::arctanh(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Natural logarithm of the elements of an array. */
    std::unique_ptr<mlx::core::array> log(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::log(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Log base 2 of the elements of an array. */
    std::unique_ptr<mlx::core::array> log2(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::log2(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Log base 10 of the elements of an array. */
    std::unique_ptr<mlx::core::array> log10(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::log10(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Natural logarithm of one plus elements in the array: `log(1 + a)`. */
    std::unique_ptr<mlx::core::array> log1p(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::log1p(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Log-add-exp of one elements in the array: `log(exp(a) + exp(b))`. */
    std::unique_ptr<mlx::core::array> logaddexp(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::logaddexp(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Element-wise logistic sigmoid of the array: `1 / (1 + exp(-x)`. */
    std::unique_ptr<mlx::core::array> sigmoid(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::sigmoid(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Computes the error function of the elements of an array. */
    std::unique_ptr<mlx::core::array> erf(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::erf(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Computes the inverse error function of the elements of an array. */
    std::unique_ptr<mlx::core::array> erfinv(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::erfinv(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Stop the flow of gradients. */
    std::unique_ptr<mlx::core::array> stop_gradient(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::stop_gradient(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Round a floating point number */
    std::unique_ptr<mlx::core::array> round(const mlx::core::array &a, int decimals, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::round(a, decimals, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Matrix-matrix multiplication. */
    std::unique_ptr<mlx::core::array> matmul(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::matmul(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Gather std::unique_ptr<mlx::core::array> entries given indices and slices */
    std::unique_ptr<mlx::core::array> gather(
        const mlx::core::array &a,
        rust::Slice<const std::unique_ptr<mlx::core::array>> indices,
        const std::vector<int> &axes,
        const std::vector<int> &slice_sizes,
        mlx_cxx::StreamOrDevice s)
    {
        auto copy_constructed_indices = std::vector<mlx::core::array>();
        for (auto &i : indices)
        {
            copy_constructed_indices.push_back(*i);
        }
        auto array = mlx::core::gather(a, copy_constructed_indices, axes, slice_sizes, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Take std::unique_ptr<mlx::core::array> slices at the given indices of the specified axis. */
    std::unique_ptr<mlx::core::array> take(
        const mlx::core::array &a,
        const mlx::core::array &indices,
        int axis,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::take(a, indices, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Take std::unique_ptr<mlx::core::array> entries at the given indices treating the std::unique_ptr<mlx::core::array> as flattened. */
    std::unique_ptr<mlx::core::array> take(const mlx::core::array &a, const mlx::core::array &indices, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::take(a, indices, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Take std::unique_ptr<mlx::core::array> entries given indices along the axis */
    std::unique_ptr<mlx::core::array> take_along_axis(
        const mlx::core::array &a,
        const mlx::core::array &indices,
        int axis,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::take_along_axis(a, indices, axis, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Scatter updates to given linear indices */
    std::unique_ptr<mlx::core::array> scatter(
        const mlx::core::array &a,
        rust::Slice<const std::unique_ptr<mlx::core::array>> indices,
        const mlx::core::array &updates,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        auto copy_constructed_indices = std::vector<mlx::core::array>();
        for (auto &i : indices)
        {
            copy_constructed_indices.push_back(*i);
        }
        auto array = mlx::core::scatter(a, copy_constructed_indices, updates, axes, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Scatter and add updates to given indices */
    std::unique_ptr<mlx::core::array> scatter_add(
        const mlx::core::array &a,
        rust::Slice<const std::unique_ptr<mlx::core::array>> indices,
        const mlx::core::array &updates,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        auto copy_constructed_indices = std::vector<mlx::core::array>();
        for (auto &i : indices)
        {
            copy_constructed_indices.push_back(*i);
        }
        auto array = mlx::core::scatter_add(a, copy_constructed_indices, updates, axes, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Scatter and prod updates to given indices */
    std::unique_ptr<mlx::core::array> scatter_prod(
        const mlx::core::array &a,
        rust::Slice<const std::unique_ptr<mlx::core::array>> indices,
        const mlx::core::array &updates,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        auto copy_constructed_indices = std::vector<mlx::core::array>();
        for (auto &i : indices)
        {
            copy_constructed_indices.push_back(*i);
        }
        auto array = mlx::core::scatter_prod(a, copy_constructed_indices, updates, axes, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Scatter and max updates to given linear indices */
    std::unique_ptr<mlx::core::array> scatter_max(
        const mlx::core::array &a,
        rust::Slice<const std::unique_ptr<mlx::core::array>> indices,
        const mlx::core::array &updates,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        auto copy_constructed_indices = std::vector<mlx::core::array>();
        for (auto &i : indices)
        {
            copy_constructed_indices.push_back(*i);
        }
        auto array = mlx::core::scatter_max(a, copy_constructed_indices, updates, axes, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Scatter and min updates to given linear indices */
    std::unique_ptr<mlx::core::array> scatter_min(
        const mlx::core::array &a,
        rust::Slice<const std::unique_ptr<mlx::core::array>> indices,
        const mlx::core::array &updates,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        auto copy_constructed_indices = std::vector<mlx::core::array>();
        for (auto &i : indices)
        {
            copy_constructed_indices.push_back(*i);
        }
        auto array = mlx::core::scatter_min(a, copy_constructed_indices, updates, axes, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Square root the elements of an array. */
    std::unique_ptr<mlx::core::array> sqrt(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::sqrt(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Square root and reciprocal the elements of an array. */
    std::unique_ptr<mlx::core::array> rsqrt(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::rsqrt(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Softmax of an array. */
    std::unique_ptr<mlx::core::array> softmax(
        const mlx::core::array &a,
        const std::vector<int> &axes,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::softmax(a, axes, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Softmax of an array. */
    std::unique_ptr<mlx::core::array> softmax(const mlx::core::array &a, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::softmax(a, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Raise elements of a to the power of b element-wise */
    std::unique_ptr<mlx::core::array> power(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::power(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Cumulative sum of an array. */
    std::unique_ptr<mlx::core::array> cumsum(
        const mlx::core::array &a,
        int axis,
        bool reverse,
        bool inclusive,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::cumsum(a, axis, reverse, inclusive, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Cumulative product of an array. */
    std::unique_ptr<mlx::core::array> cumprod(
        const mlx::core::array &a,
        int axis,
        bool reverse,
        bool inclusive,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::cumprod(a, axis, reverse, inclusive, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Cumulative max of an array. */
    std::unique_ptr<mlx::core::array> cummax(
        const mlx::core::array &a,
        int axis,
        bool reverse,
        bool inclusive,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::cummax(a, axis, reverse, inclusive, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Cumulative min of an array. */
    std::unique_ptr<mlx::core::array> cummin(
        const mlx::core::array &a,
        int axis,
        bool reverse,
        bool inclusive,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::cummin(a, axis, reverse, inclusive, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Convolution operations */

    /** 1D convolution with a filter */
    std::unique_ptr<mlx::core::array> conv1d(
        const mlx::core::array &input,
        const mlx::core::array &weight,
        int stride,
        int padding,
        int dilation,
        int groups,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::conv1d(input, weight, stride, padding, dilation, groups, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** 2D convolution with a filter */
    std::unique_ptr<mlx::core::array> conv2d(
        const mlx::core::array &input,
        const mlx::core::array &weight,
        const std::array<int, 2> &stride,
        const std::array<int, 2> &padding,
        const std::array<int, 2> &dilation,
        int groups,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::conv2d(input, weight, stride, padding, dilation, groups, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Quantized matmul multiplies x with a quantized matrix w*/
    std::unique_ptr<mlx::core::array> quantized_matmul(
        const mlx::core::array &x,
        const mlx::core::array &w,
        const mlx::core::array &scales,
        const mlx::core::array &biases,
        bool transpose,
        int group_size,
        int bits,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::quantized_matmul(x, w, scales, biases, transpose, group_size, bits, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Quantize a matrix along its last axis */
    std::array<std::unique_ptr<mlx::core::array>, 3> quantize(
        const mlx::core::array &w,
        int group_size,
        int bits,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::quantize(w, group_size, bits, s.to_variant());
        return std::array<std::unique_ptr<mlx::core::array>, 3>{
            std::make_unique<mlx::core::array>(std::get<0>(array)),
            std::make_unique<mlx::core::array>(std::get<1>(array)),
            std::make_unique<mlx::core::array>(std::get<2>(array))};
    }

    /** Dequantize a matrix produced by quantize() */
    std::unique_ptr<mlx::core::array> dequantize(
        const mlx::core::array &w,
        const mlx::core::array &scales,
        const mlx::core::array &biases,
        int group_size,
        int bits,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::dequantize(w, scales, biases, group_size, bits, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** TensorDot returns a contraction of a and b over multiple dimensions. */
    std::unique_ptr<mlx::core::array> tensordot(
        const mlx::core::array &a,
        const mlx::core::array &b,
        const int dims,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::tensordot(a, b, dims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> tensordot(
        const mlx::core::array &a,
        const mlx::core::array &b,
        const std::array<std::unique_ptr<std::vector<int>>, 2> &dims,
        mlx_cxx::StreamOrDevice s)
    {
        auto pair_dims = std::pair<std::vector<int>, std::vector<int>>(*dims[0], *dims[1]);
        auto array = mlx::core::tensordot(a, b, pair_dims, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Compute the outer product of two vectors. */
    std::unique_ptr<mlx::core::array> outer(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::outer(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Compute the inner product of two vectors. */
    std::unique_ptr<mlx::core::array> inner(const mlx::core::array &a, const mlx::core::array &b, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::inner(a, b, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Compute D = beta * C + alpha * (A @ B) */
    std::unique_ptr<mlx::core::array> addmm(
        std::unique_ptr<mlx::core::array> c,
        std::unique_ptr<mlx::core::array> a,
        std::unique_ptr<mlx::core::array> b,
        const float &alpha,
        const float &beta,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::addmm(*c, *a, *b, alpha, beta, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Extract a diagonal or construct a diagonal array */
    std::unique_ptr<mlx::core::array> diagonal(
        const mlx::core::array& a,
        int offset,
        int axis1,
        int axis2,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::diagonal(a, offset, axis1, axis2, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /** Extract diagonal from a 2d array or create a diagonal matrix. */
    std::unique_ptr<mlx::core::array> diag(const mlx::core::array& a, int k, mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::diag(a, k, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    /**
     * Implements the identity function but allows injecting dependencies to other
     * arrays. This ensures that these other arrays will have been computed
     * when the outputs of this function are computed.
     */
    std::unique_ptr<std::vector<mlx::core::array>> depends(
        const std::vector<mlx::core::array>& inputs,
        const std::vector<mlx::core::array>& dependencies)
    {
        auto arrays = mlx::core::depends(inputs, dependencies);
        return std::make_unique<std::vector<mlx::core::array>>(arrays);
    }
}