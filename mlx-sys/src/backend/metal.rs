#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/backend/metal/metal.h");
        include!("mlx-cxx/backend/metal/metal.hpp");

        #[namespace = "mlx::core::metal"]
        fn is_available() -> bool;

        #[namespace = "mlx::core::metal"]
        fn get_active_memory() -> usize;

        #[namespace = "mlx::core::metal"]
        fn get_peak_memory() -> usize;

        #[namespace = "mlx::core::metal"]
        fn get_cache_memory() -> usize;

        #[namespace = "mlx::core::metal"]
        fn set_memory_limit(limit: usize, relaxed: bool) -> usize;

        #[namespace = "mlx::core::metal"]
        fn set_cache_limit(limit: usize) -> usize;

        #[namespace = "mlx_cxx::metal"]
        fn start_capture(path: UniquePtr<CxxString>) -> bool;

        #[namespace = "mlx::core::metal"]
        fn stop_capture();

        // TODO: should these be exported? `new_stream()`, `new_scoped_memory_pool()`, and
        // `make_task()`
    }
}

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn test_is_available() {
//         #[cfg(feature = "metal")]
//         assert!(super::ffi::is_available());

//         #[cfg(not(feature = "metal"))]
//         assert!(!super::ffi::is_available());
//     }

//     #[test]
//     fn test_cache_enabled() {
//         let _ = super::ffi::cache_enabled();
//     }

//     #[test]
//     fn test_set_cache_enabled() {
//         super::ffi::set_cache_enabled(true);
//         super::ffi::set_cache_enabled(false);
//     }
// }
