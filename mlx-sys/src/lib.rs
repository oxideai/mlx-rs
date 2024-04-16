#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub fn is_metal_available() -> bool {
    unsafe {
        mlx_metal_is_available()
    }
}
