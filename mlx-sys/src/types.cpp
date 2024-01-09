#include "mlx-cxx/types.hpp"
#include "mlx-sys/src/types/float16.rs.h"
#include "mlx-sys/src/types/bfloat16.rs.h"
#include "mlx-sys/src/types/complex64.rs.h"

#include "mlx/dtype.h"

namespace mlx_cxx {
    // mlx::core::complex64_t complex64(float v, float u) {
    //     return mlx::core::complex64_t(v, u);
    // }

    // mlx::core::float16_t bits_to_float16_t(uint16_t bits) {
    //     static_assert(sizeof(mlx::core::float16_t) == sizeof(uint16_t), "float16_t is not 16 bits");
    //     mlx::core::float16_t out;
    //     std::memcpy(&out, &bits, sizeof(uint16_t));
    //     return out;
    // }

    mlx::core::float16_t f16_to_float16_t(mlx_cxx::float16_t value) {
        static_assert(sizeof(mlx::core::float16_t) == sizeof(uint16_t), "Size of float16_t is not equal to size of uint16_t");
        uint16_t bits = f16_to_bits(value);
        mlx::core::float16_t out;
        std::memcpy(&out, &bits, sizeof(uint16_t));
        return out;
    }

    // TODO: this is only a test. Remove later
    uint16_t test_f16_to_bits(mlx_cxx::float16_t value) {
        uint16_t bits = f16_to_bits(value);
        return bits;
    }

    mlx::core::bfloat16_t bf16_to_bfloat16_t(mlx_cxx::bfloat16_t value) {
        static_assert(sizeof(mlx::core::bfloat16_t) == sizeof(uint16_t), "Size of bfloat16_t is not equal to size of uint16_t");
        uint16_t bits = bf16_to_bits(value);
        mlx::core::bfloat16_t out;
        std::memcpy(&out, &bits, sizeof(uint16_t));
        return out;
    }

    uint16_t test_bf16_to_bits(mlx_cxx::bfloat16_t value) {
        uint16_t bits = bf16_to_bits(value);
        return bits;
    }

    mlx::core::complex64_t c64_to_complex64_t(complex64_t value) {
        float re = value.real();
        float im = value.imag();
        mlx::core::complex64_t out(re, im);
        return out;
    }
}

namespace mlx::core {
    template<>
    TypeToDtype<mlx_cxx::float16_t>::operator Dtype() {
        return mlx::core::float16;
    }

    template<>
    TypeToDtype<mlx_cxx::bfloat16_t>::operator Dtype() {
        return mlx::core::bfloat16;
    }
}