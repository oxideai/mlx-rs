#pragma once

#include "mlx/array.h"
#include "mlx/compile.h"

#include "mlx-cxx/transforms.hpp"

#include "mlx-sys/src/compat.rs.h"

namespace mlx_cxx {
    std::unique_ptr<CxxMultiaryFn> compile(const CxxMultiaryFn &fun);

    std::unique_ptr<CxxMultiaryFn> compile(const MultiaryFn *fun);
}