macro_rules! try_catch_c_ptr_expr {
    ($expr:expr) => {{
        if !$crate::error::is_mlx_error_handler_set() {
            $crate::error::setup_mlx_error_handler();
        }

        let c_ptr = $expr;
        if c_ptr.is_null() {
            // SAFETY: there must be an error if the pointer is null
            return Err($crate::error::get_and_clear_last_mlx_error()
                .expect("A null pointer was returned, but no error was set."));
        }
        c_ptr
    }};
}

/// This is intended to be used on expressions that contains a closure that may
/// return an error.
macro_rules! try_catch_mlx_closure_error {
    ($expr:expr) => {{
        if !$crate::error::is_mlx_error_handler_set() {
            $crate::error::setup_mlx_error_handler();
        }

        let c_ptr = $expr;

        // Always check for error, even if the pointer is not null
        if let Some(e) = $crate::error::get_and_clear_last_mlx_error()
            .or_else(|| $crate::error::take_last_mlx_closure_error()) 
        {
            return Err(e);
        }
        if c_ptr.is_null() {
            panic!("A null pointer was returned, but no error was set.");
        }
        c_ptr
    }};
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
