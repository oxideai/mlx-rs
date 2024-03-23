use cxx::{memory::UniquePtrTarget, UniquePtr};
use mlx_sys::compat::CompatFn;

pub enum Func<F> 
where
    F: CompatFn,
    F::CxxFn: UniquePtrTarget,
{
    Rust(F),
    Cxx(UniquePtr<F::CxxFn>),
} 