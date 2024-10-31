#include <functional>

#include "mlx/c/object.h"
#include "mlx/c/string.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/closure.h"
#include "mlx/c/private/string.h"
#include "mlx/c/private/utils.h"
#include "mlx/c/private/vector.h"

#include "closure.h"
#include "result.h"


extern "C" mlx_closure mlx_fallible_closure_new_with_payload(
    mlx_vector_array_result (*fun)(const mlx_vector_array, void*),
    void* payload,
    void (*dtor)(void*)
) {
    auto cpp_payload = std::shared_ptr<void>(payload, dtor);
    auto cpp_closure =
      [fun, cpp_payload](const std::vector<mlx::core::array>& cpp_input) {
        auto input = new mlx_vector_array_(cpp_input);
        auto res = fun(input, cpp_payload.get());
        mlx_free(input);
        if (mlx_vector_array_result_is_err(&res)) {
            auto err = mlx_vector_array_result_into_err(res);
            std::string msg = std::move(err->ctx);
            mlx_free(err);
            throw std::runtime_error(msg);
        }
        auto c_ok = mlx_vector_array_result_into_ok(res);
        auto cpp_res = c_ok->ctx;
        mlx_free(c_ok);
        return cpp_res;
    };
    MLX_TRY_CATCH(return new mlx_closure_(cpp_closure), return nullptr);
}