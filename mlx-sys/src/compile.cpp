#include "mlx-cxx/compile.hpp"

namespace mlx_cxx {
    std::unique_ptr<CxxMultiaryFn> compile(const MultiaryFn *fun)
    {
        return mlx_cxx::compile(make_multiary_fn(fun));
    }

    std::unique_ptr<CxxMultiaryFn> compile(const MultiaryFn *fun)
    {
        return mlx_cxx::compile(make_multiary_fn(fun));
    }
}