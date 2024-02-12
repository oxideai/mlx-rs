#include "mlx-cxx/transforms.hpp"
#include "mlx-cxx/compile.hpp"

namespace mlx_cxx {
    std::unique_ptr<CxxMultiaryFn> compile(const CxxMultiaryFn &fun) {
        throw std::runtime_error("Not implemented");
    }

    std::unique_ptr<CxxMultiaryFn> compile(const MultiaryFn *fun) {
        throw std::runtime_error("Not implemented");
    }
}