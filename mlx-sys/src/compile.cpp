#include "mlx-cxx/transforms.hpp"
#include "mlx-cxx/compile.hpp"

namespace mlx_cxx {
    std::unique_ptr<CxxMultiaryFn> compile(const CxxMultiaryFn &fun, bool shapeless) {
        auto ret = mlx::core::compile(fun, shapeless);
        return std::make_unique<CxxMultiaryFn>(std::move(ret));
    }
}