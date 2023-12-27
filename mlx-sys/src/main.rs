#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("mlx/mlx.h");

        fn say_something();
    }
}

fn main() {
    ffi::say_something();
}