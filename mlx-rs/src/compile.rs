use cxx::Exception;
use mlx_sys::compat::MultiaryFn;
// Re-export
pub use mlx_sys::compile::ffi::{CompileMode, disable_compile, enable_compile, set_compile_mode};

use crate::function::Func;

pub fn compile(fun: &Func<MultiaryFn>, shapeless: bool) -> Result<Func<MultiaryFn>, Exception> {
    let ret = match fun {
        Func::Rust(f) => {
            let raw_ptr = f as *const MultiaryFn;
            unsafe {mlx_sys::compile::compat::ffi::compile(raw_ptr, shapeless)?}
        },
        Func::Cxx(f) => mlx_sys::compile::ffi::compile(&f, shapeless)?,
    };

    Ok(Func::Cxx(ret))
}