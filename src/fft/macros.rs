macro_rules! fft {
    ($a:tt) => {
        $crate::fft::fft($a, None, None)
    };
}