#include <functional>

#include "closure.h"

mlx_vector_array trampoline(const mlx_vector_array input, void* cls) {
    auto cpp_cls = static_cast<std::function<mlx_vector_array(const mlx_vector_array)>*>(cls);
    return (*cpp_cls)(input);
}

extern "C" mlx_closure mlx_fallible_closure_new_with_payload(
    mlx_vector_array_result (*fun)(const mlx_vector_array, void*),
    void* payload,
    void (*dtor)(void*)) {
    auto cls = [fun, payload](const mlx_vector_array input) {
        auto c_res = fun(input, payload);
        if (mlx_vector_array_result_is_err(&c_res)) {
            mlx_string err = mlx_vector_array_result_into_err(c_res);
            throw std::runtime_error(mlx_string_data(err));
        }
        return mlx_vector_array_result_into_ok(c_res);
    };

    return mlx_closure_new_with_payload(trampoline, &cls, dtor);
}