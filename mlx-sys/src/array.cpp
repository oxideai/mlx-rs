#include "mlx-cxx/mlx_cxx.hpp"
#include "mlx-cxx/array.hpp"
#include "mlx-cxx/types.hpp"
#include "mlx-sys/src/types/float16.rs.h"

#include "mlx/types/half_types.h"

namespace mlx_cxx {
    // std::unique_ptr<array> array_new_bool(bool value) {
    //     return mlx_cxx::new_unique<array>(value);
    // }

    std::unique_ptr<array> array_new_f16(mlx_cxx::f16 value) {
        mlx::core::float16_t value2 = mlx_cxx::f16_to_float16_t(value);
        return mlx_cxx::new_unique<array>(value2);
    }
}
