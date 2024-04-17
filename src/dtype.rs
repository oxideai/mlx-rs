
/// Array element type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl TryFrom<u32> for Dtype {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            mlx_sys::mlx_array_dtype__MLX_BOOL => Ok(Dtype::Bool),
            mlx_sys::mlx_array_dtype__MLX_UINT8 => Ok(Dtype::Uint8),
            mlx_sys::mlx_array_dtype__MLX_UINT16 => Ok(Dtype::Uint16),
            mlx_sys::mlx_array_dtype__MLX_UINT32 => Ok(Dtype::Uint32),
            mlx_sys::mlx_array_dtype__MLX_UINT64 => Ok(Dtype::Uint64),
            mlx_sys::mlx_array_dtype__MLX_INT8 => Ok(Dtype::Int8),
            mlx_sys::mlx_array_dtype__MLX_INT16 => Ok(Dtype::Int16),
            mlx_sys::mlx_array_dtype__MLX_INT32 => Ok(Dtype::Int32),
            mlx_sys::mlx_array_dtype__MLX_INT64 => Ok(Dtype::Int64),
            mlx_sys::mlx_array_dtype__MLX_FLOAT16 => Ok(Dtype::Float16),
            mlx_sys::mlx_array_dtype__MLX_FLOAT32 => Ok(Dtype::Float32),
            mlx_sys::mlx_array_dtype__MLX_BFLOAT16 => Ok(Dtype::Bfloat16),
            mlx_sys::mlx_array_dtype__MLX_COMPLEX64 => Ok(Dtype::Complex64),
            _ => Err(value),
        }
    }
}
