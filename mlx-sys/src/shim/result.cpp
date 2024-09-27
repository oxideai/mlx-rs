#include "result.h"

extern "C" mlx_vector_array_result mlx_vector_array_result_new_ok(
    mlx_vector_array ok) {
  mlx_vector_array_result result;
  result.tag = mlx_result_tag_ok;
  result.ok = ok;
  return result;
}

extern "C" mlx_vector_array_result mlx_vector_array_result_new_err(
    mlx_string err) {
  mlx_vector_array_result result;
  result.tag = mlx_result_tag_err;
  result.err = err;
  return result;
}

extern "C" bool mlx_vector_array_result_is_ok(mlx_vector_array_result* result) {
  return result->tag == mlx_result_tag_ok;
}

extern "C" bool mlx_vector_array_result_is_err(
    mlx_vector_array_result* result) {
  return result->tag == mlx_result_tag_err;
}

extern "C" mlx_result_tag mlx_vector_array_result_get_tag(
    mlx_vector_array_result* result) {
  return result->tag;
}

mlx_vector_array mlx_vector_array_result_into_ok(
    mlx_vector_array_result result) {
  return result.ok;
}

extern "C" mlx_string mlx_vector_array_result_into_err(
    mlx_vector_array_result result) {
  return result.err;
}