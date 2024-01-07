#pragma once

#include <cstdint>

#include "mlx/types/half_types.h"
#include "mlx/types/complex.h"


namespace mlx_cxx {
    struct f16;
    struct bf16;
    struct c64;

    mlx::core::float16_t f16_to_float16_t(f16 value);

    uint16_t test_f16_to_bits(f16 value);

    mlx::core::bfloat16_t bf16_to_bfloat16_t(bf16 value);

    uint16_t test_bf16_to_bits(bf16 value);

    mlx::core::complex64_t c64_to_complex64_t(c64 value);
}