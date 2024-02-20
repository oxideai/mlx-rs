#[cxx::bridge]
pub mod ffi {
    #[namespace = "mlx::core"]
    #[repr(i32)]
    enum CompileMode {
        #[cxx_name = "disabled"]
        Disabled,

        #[cxx_name = "no_simplify"]
        NoSimplify,

        #[cxx_name = "no_fuse"]
        NoFuse,

        #[cxx_name = "enabled"]
        Enabled,
    }

    unsafe extern "C++" {
        include!("mlx/compile.h");
        include!("mlx-cxx/transforms.hpp");
        include!("mlx-cxx/compile.hpp");

        #[namespace = "mlx::core"]
        type CompileMode;

        #[namespace = "mlx_cxx"]
        type CxxMultiaryFn = crate::transforms::ffi::CxxMultiaryFn;

        #[namespace = "mlx_cxx"]
        fn compile(fun: &CxxMultiaryFn) -> Result<UniquePtr<CxxMultiaryFn>>;

        #[namespace = "mlx::core"]
        fn disable_compile();

        #[namespace = "mlx::core"]
        fn enable_compile();

        #[namespace = "mlx::core"]
        fn set_compile_mode(mode: CompileMode);
    }
}

pub mod compat {
    pub mod ffi {
        pub use crate::compat::{ffi::compile, ffi::CxxMultiaryFn, MultiaryFn};
    }
}
