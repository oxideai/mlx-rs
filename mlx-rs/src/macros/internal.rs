macro_rules! check_status {
    ($status:expr, $dtor:expr) => {
        // TODO: should we just add some kind of init function and ask user to call it at the
        // beginning of the program?
        if !$crate::error::is_mlx_error_handler_set() {
            $crate::error::setup_mlx_error_handler();
        }

        if $status != crate::utils::SUCCESS {
            $dtor;
            return Err($crate::error::get_and_clear_last_mlx_error()
                .expect("A non-success status was returned, but no error was set.")
                .into());
        }
    };
}

/// See `assertEqual` in the swift binding tests
#[allow(unused_macros)]
macro_rules! assert_array_all_close {
    ($a:tt, $b:tt) => {
        let _b: Array = $b.into();
        let assert = $a.all_close(&_b, None, None, None).unwrap();
        assert!(assert.item::<bool>());
    };
}

macro_rules! debug_panic {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            panic!($($arg)*);
        }
    };
}
