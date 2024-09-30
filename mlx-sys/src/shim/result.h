#ifndef MLC_X_SHIM_RESULT_H
#define MLC_X_SHIM_RESULT_H

#include "mlx/c/array.h"
#include "mlx/c/string.h"

#ifdef __cplusplus
#include <stdexcept>

extern "C" {
#endif

typedef enum {
  mlx_result_tag_ok,
  mlx_result_tag_err,
} mlx_result_tag;

typedef struct {
  mlx_result_tag tag;
  union {
    mlx_vector_array ok;
    mlx_string err;
  };
} mlx_vector_array_result;

mlx_vector_array_result mlx_vector_array_result_new_ok(mlx_vector_array ok);
mlx_vector_array_result mlx_vector_array_result_new_err(mlx_string err);
mlx_result_tag mlx_vector_array_result_get_tag(mlx_vector_array_result* result);
bool mlx_vector_array_result_is_ok(mlx_vector_array_result* result);
bool mlx_vector_array_result_is_err(mlx_vector_array_result* result);
mlx_vector_array mlx_vector_array_result_into_ok(
    mlx_vector_array_result result);
mlx_string mlx_vector_array_result_into_err(mlx_vector_array_result result);

#ifdef __cplusplus
}
#endif

#endif // MLC_X_SHIM_RESULT_H