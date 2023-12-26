use ffi::foobar;


#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("mlx/mlx.h");

        fn foobar();
    }
}

fn main() {
    ffi::foobar();
}