/// Array element type
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, num_enum::IntoPrimitive, num_enum::TryFromPrimitive,
)]
#[repr(u32)]
pub enum Dtype {
    Bool = mlx_sys::mlx_array_dtype__MLX_BOOL,
    Uint8 = mlx_sys::mlx_array_dtype__MLX_UINT8,
    Uint16 = mlx_sys::mlx_array_dtype__MLX_UINT16,
    Uint32 = mlx_sys::mlx_array_dtype__MLX_UINT32,
    Uint64 = mlx_sys::mlx_array_dtype__MLX_UINT64,
    Int8 = mlx_sys::mlx_array_dtype__MLX_INT8,
    Int16 = mlx_sys::mlx_array_dtype__MLX_INT16,
    Int32 = mlx_sys::mlx_array_dtype__MLX_INT32,
    Int64 = mlx_sys::mlx_array_dtype__MLX_INT64,
    Float16 = mlx_sys::mlx_array_dtype__MLX_FLOAT16,
    Float32 = mlx_sys::mlx_array_dtype__MLX_FLOAT32,
    Bfloat16 = mlx_sys::mlx_array_dtype__MLX_BFLOAT16,
    Complex64 = mlx_sys::mlx_array_dtype__MLX_COMPLEX64,
}
