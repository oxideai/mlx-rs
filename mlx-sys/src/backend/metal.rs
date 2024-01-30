#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/backend/metal/metal.h");

        #[namespace = "mlx::core::metal"]
        fn is_available() -> bool;

        #[namespace = "mlx::core::metal"]
        fn cache_enabled() -> bool;

        #[namespace = "mlx::core::metal"]
        fn set_cache_enabled(enabled: bool);

        // TODO: it seems like other function are not supposed to be
        // exposed to the user.
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
