cmake_minimum_required(VERSION 3.25)
project(bump_canaries LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(MLX_C_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
add_subdirectory("${MLX_C_SOURCE}" mlxc)
# mlx-c disables upstream tests; add them only in this isolated build.
set(PROJECT_SOURCE_DIR "${mlx_SOURCE_DIR}")
include(CTest)
add_subdirectory("${mlx_SOURCE_DIR}/tests" upstream-tests)
