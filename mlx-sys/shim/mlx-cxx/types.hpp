#pragma once

#include <cstdint>

#include "mlx/types/half_types.h"
#include "mlx/types/complex.h"


namespace mlx_cxx {
    struct float16_t;
    struct bf16;
    struct c64;

    mlx::core::float16_t f16_to_float16_t(float16_t value);

    uint16_t test_f16_to_bits(float16_t value);

    mlx::core::bfloat16_t bf16_to_bfloat16_t(bf16 value);

    uint16_t test_bf16_to_bits(bf16 value);

    mlx::core::complex64_t c64_to_complex64_t(c64 value);
}