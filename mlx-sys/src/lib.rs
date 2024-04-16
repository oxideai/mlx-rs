#[link(name="mlxc", kind="static")]
extern {
    fn mlx_metal_is_available() -> bool;
}

pub fn is_metal_available() -> bool {
    unsafe {
        mlx_metal_is_available()
    }
}
