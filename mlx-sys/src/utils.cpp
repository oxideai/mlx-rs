#include "mlx/utils.h"
#include "mlx/array.h"
#include "mlx-cxx/utils.hpp"

namespace mlx_cxx {
    mlx::core::Dtype result_type(rust::Slice<const std::unique_ptr<mlx::core::array>> arrays) {
        // Create a vector of arrays from the slice using the copy constructor
        std::vector<mlx::core::array> copy_constructed_arrays;
        for (auto& array : arrays) {
            copy_constructed_arrays.push_back(*array);
        }
        return mlx::core::result_type(copy_constructed_arrays);
    }

    std::unique_ptr<std::vector<int>> broadcast_shapes(
        const std::vector<int>& s1,
        const std::vector<int>& s2) {
        return std::make_unique<std::vector<int>>(mlx::core::broadcast_shapes(s1, s2));
    }

    bool is_same_shape(rust::Slice<const std::unique_ptr<mlx::core::array>> arrays) {
        // Create a vector of arrays from the slice using the copy constructor
        std::vector<mlx::core::array> copy_constructed_arrays;
        for (auto& array : arrays) {
            copy_constructed_arrays.push_back(*array);
        }
        return mlx::core::is_same_shape(copy_constructed_arrays);
    }
}