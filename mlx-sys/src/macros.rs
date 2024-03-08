#[macro_export]
macro_rules! cxx_vec {
    () => {
        ::mlx_sys::cxx::CxxVector::new()
    };
    ($($x:expr),*) => {
        ::mlx_sys::utils::IntoCxxVector::into_cxx_vector([$($x),*])
    };
}
