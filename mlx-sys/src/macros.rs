#[macro_export]
macro_rules! cxx_vec {
    () => {
        cxx::CxxVector::new()
    };
    ($($x:expr),*) => {
        {
            let mut v = cxx::CxxVector::new();
            $(
                v.pin_mut().push($x);
            )*
            v
        }
    };
}
