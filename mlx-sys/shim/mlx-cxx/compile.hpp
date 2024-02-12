#pragma once

#include "mlx-cxx/functions.hpp"

namespace mlx_cxx {
    std::unique_ptr<CxxMultiaryFn> compile(const CxxMultiaryFn &fun);

    // std::unique_ptr<CxxMultiaryFn> compile(const MultiaryFn *fun);
}