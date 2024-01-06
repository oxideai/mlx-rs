#pragma once

#include <memory>

namespace mlx_cxx {
    // Generic template constructor
    template <typename T, typename... Args> std::unique_ptr<T> new_unique(Args... args) {
        return std::unique_ptr<T>(new T(args...));
    }
}
