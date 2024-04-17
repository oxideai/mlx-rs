pub struct Array {
    ctx: mlx_sys::mlx_array,
}

impl Array {
    pub(crate) fn new(ctx: mlx_sys::mlx_array) -> Array {
        Array { ctx }
    }

    pub(crate) fn ctx(&self) -> mlx_sys::mlx_array {
        self.ctx
    }
}
