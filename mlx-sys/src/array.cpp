#include "mlx-cxx/mlx_cxx.hpp"
#include "mlx-cxx/array.hpp"
#include "mlx-cxx/types.hpp"

#include "mlx-sys/src/types/float16.rs.h"
#include "mlx-sys/src/types/bfloat16.rs.h"
#include "mlx-sys/src/types/complex64.rs.h"

#include "mlx/types/half_types.h"
#include "mlx/types/complex.h"

namespace mlx_cxx {
    // std::unique_ptr<array> array_new_bool(bool value) {
    //     return mlx_cxx::new_unique<array>(value);
    // }

    std::unique_ptr<array> array_new_f16(mlx_cxx::f16 value) {
        mlx::core::float16_t value2 = mlx_cxx::f16_to_float16_t(value);
        return mlx_cxx::new_unique<array>(value2);
    }

    std::unique_ptr<array> array_new_bf16(mlx_cxx::bf16 value) {
        mlx::core::bfloat16_t value2 = mlx_cxx::bf16_to_bfloat16_t(value);
        return mlx_cxx::new_unique<array>(value2);
    }

    std::unique_ptr<array> array_new_c64(mlx_cxx::c64 value) {
        mlx::core::complex64_t value2 = mlx_cxx::c64_to_complex64_t(value);
        return mlx_cxx::new_unique<array>(value2);
    }
}
