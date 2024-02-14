#pragma once

#include "mlx/dtype.h"
#include "mlx/utils.h"

#include "rust/cxx.h"

namespace mlx_cxx {
    // TODO: add binding to print format?

    mlx::core::Dtype result_type(rust::Slice<const std::unique_ptr<mlx::core::array>> arrays);

    std::unique_ptr<std::vector<int>> broadcast_shapes(
        const std::vector<int>& s1,
        const std::vector<int>& s2);

    bool is_same_shape(rust::Slice<const std::unique_ptr<mlx::core::array>> arrays);

    template<typename T>
    void push_opaque(
        std::vector<T>& vec,
        std::unique_ptr<T> item);

    template<typename T>
    std::unique_ptr<T> pop_opaque(
        std::vector<T>& vec);

    template<typename T>
    std::unique_ptr<std::vector<T>> std_vec_from_slice(
        rust::Slice<const std::unique_ptr<T>> slice);
}