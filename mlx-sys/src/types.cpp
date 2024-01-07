#include "mlx-cxx/types.hpp"
#include "mlx-sys/src/types/float16.rs.h"

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

    // TODO: this is only a test. Remove later
    uint16_t cxx_f16_to_bits(f16 value) {
        uint16_t bits = f16_to_bits(value);
        return bits;
    }

    mlx::core::float16_t f16_to_float16_t(f16 value) {
        static_assert(sizeof(mlx::core::float16_t) == sizeof(uint16_t), "Size of float16_t is not equal to size of uint16_t");
        uint16_t bits = f16_to_bits(value);
        mlx::core::float16_t out;
        std::memcpy(&out, &bits, sizeof(uint16_t));
        return out;
    }
}

namespace mlx::core {
    template<>
    TypeToDtype<mlx_cxx::f16>::operator Dtype() {
        return mlx::core::float16;
    }
}