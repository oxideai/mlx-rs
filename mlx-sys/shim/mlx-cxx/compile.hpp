#pragma once

#include "mlx/compile.h"

#include "mlx-cxx/functions.hpp"

#include "mlx-sys/src/compat.rs.h"

namespace mlx_cxx {
    std::unique_ptr<CxxMultiaryFn> compile(const CxxMultiaryFn &fun);
}