#ifndef MLX_C_SHIM_CLOSURE_H
#define MLX_C_SHIM_CLOSURE_H

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/vector.h"

#include "result.h"

#ifdef __cplusplus
extern "C" {
#endif

mlx_closure mlx_fallible_closure_new_with_payload(
    mlx_vector_array_result (*fun)(const mlx_vector_array, void*),
    void* payload,
    void (*dtor)(void*));

#ifdef __cplusplus
}
#endif

#endif // MLX_C_SHIM_CLOSURE_H