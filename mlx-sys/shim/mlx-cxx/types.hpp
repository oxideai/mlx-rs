#pragma once
// #include "mlx/types/complex.h"
#include "mlx/types/half_types.h"

#include <cstdint>

namespace mlx_cxx {
    struct f16;

    uint16_t cxx_f16_to_bits(f16 value);

    mlx::core::float16_t f16_to_float16_t(f16 value);
}