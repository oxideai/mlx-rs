#include <functional>

#include "mlx/c/object.h"
#include "mlx/c/string.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/closure.h"
#include "mlx/c/private/string.h"
#include "mlx/c/private/utils.h"

#include "closure.h"
#include "result.h"

extern "C" mlx_closure mlx_fallible_closure_new_with_payload(
    mlx_vector_array_result (*fun)(const mlx_vector_array, void*),
    void* payload,
    void (*dtor)(void*)
) {
    auto cpp_payload = std::shared_ptr<void>(payload, dtor);
    auto cpp_closure = [fun, cpp_payload](const std::vector<mlx::core::array>& input) {
        auto c_input = new mlx_vector_array_(input);
        auto c_res = fun(c_input, cpp_payload.get());
        mlx_free(c_input);
        if (mlx_vector_array_result_is_err(&c_res)) {
            auto err = mlx_vector_array_result_into_err(c_res);
            std::string msg = std::move(err->ctx);
            mlx_free(err);
            throw std::runtime_error(msg);
        }
        auto c_ok = mlx_vector_array_result_into_ok(c_res);
        auto res = c_ok->ctx;
        mlx_free(c_ok);
        return res;
    };
    MLX_TRY_CATCH(return new mlx_closure_(cpp_closure), return nullptr);
}