use cxx::{CxxVector, UniquePtr};

use crate::array::ffi::array;

pub trait Function<Args> {
    type Output;

    fn execute(&self, args: Args) -> Self::Output;
}

impl<Args, F, R> Function<Args> for F
where
    F: Fn(Args) -> R,
{
    type Output = R;

    fn execute(&self, args: Args) -> Self::Output {
        self(args)
    }
}

#[repr(transparent)]
pub struct UnaryFn(pub Box<dyn for<'a> Function<&'a array, Output = UniquePtr<array>> + 'static>);

impl<F> From<F> for UnaryFn
where
    F: for<'a> Function<&'a array, Output = UniquePtr<array>> + 'static,
{
    fn from(f: F) -> Self {
        Self(Box::new(f))
    }
}

pub fn execute_unary_fn(f: &UnaryFn, args: &array) -> UniquePtr<array> {
    f.0.execute(args)
}

#[repr(transparent)]
pub struct MultiaryFn(pub Box<dyn for<'a> Function<&'a CxxVector<array>, Output = UniquePtr<CxxVector<array>>> + 'static>);

impl<F> From<F> for MultiaryFn
where
    F: for<'a> Function<&'a CxxVector<array>, Output = UniquePtr<CxxVector<array>>> + 'static,
{
    fn from(f: F) -> Self {
        Self(Box::new(f))
    }
}

pub fn execute_multiary_fn(f: &MultiaryFn, args: &CxxVector<array>) -> UniquePtr<CxxVector<array>> {
    f.0.execute(args)
}

#[repr(transparent)]
pub struct MultiInputSingleOutputFn(
    pub Box<dyn for<'a> Function<&'a CxxVector<array>, Output=UniquePtr<array>> + 'static>
);

impl<F> From<F> for MultiInputSingleOutputFn
where
    F: for<'a> Function<&'a CxxVector<array>, Output=UniquePtr<array>> + 'static,
{
    fn from(f: F) -> Self {
        Self(Box::new(f))
    }
}

fn execute_multi_input_single_output_fn(
    f: &MultiInputSingleOutputFn,
    args: &CxxVector<array>,
) -> UniquePtr<array> {
    f.0.execute(args)
}

#[repr(transparent)]
pub struct PairInputSingleOutputFn(
    pub Box<dyn for<'a> Function<[&'a array; 2], Output=UniquePtr<array>> + 'static>
);

impl<F> From<F> for PairInputSingleOutputFn
where
    F: for<'a> Function<[&'a array; 2], Output=UniquePtr<array>> + 'static,
{
    fn from(f: F) -> Self {
        Self(Box::new(f))
    }
}

// This runs into problem with the c++ binding if we use [&array; 2] as the input type
fn execute_pair_input_single_output_fn(
    f: &PairInputSingleOutputFn,
    first: &array,
    second: &array,
) -> UniquePtr<array> {
    f.0.execute([first, second])
}

// TODO: change visibility and then re-export
#[cxx::bridge]
pub mod ffi {
    extern "C++" {
        include!("mlx/array.h");

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;
    }

    extern "Rust" {
        #[namespace = "mlx_cxx"]
        type UnaryFn;

        #[namespace = "mlx_cxx"]
        type MultiaryFn;

        #[namespace = "mlx_cxx"]
        type MultiInputSingleOutputFn;

        #[namespace = "mlx_cxx"]
        type PairInputSingleOutputFn;

        #[namespace = "mlx_cxx"]
        fn execute_unary_fn(f: &UnaryFn, x: &array) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn execute_multiary_fn(
            f: &MultiaryFn,
            xs: &CxxVector<array>,
        ) -> UniquePtr<CxxVector<array>>;

        #[namespace = "mlx_cxx"]
        fn execute_multi_input_single_output_fn(
            f: &MultiInputSingleOutputFn,
            xs: &CxxVector<array>,
        ) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn execute_pair_input_single_output_fn(
            f: &PairInputSingleOutputFn,
            first: &array,
            second: &array,
        ) -> UniquePtr<array>;
    }

    unsafe extern "C++" {
        include!("mlx-cxx/transforms.hpp");

        // TODO: This is for test only. Remove later
        #[namespace = "mlx_cxx"]
        fn accept_rust_unary_fn(f: &UnaryFn) -> i32;

        #[namespace = "mlx_cxx"]
        #[rust_name = "vjp_unary_fn"]
        unsafe fn vjp(f: *const UnaryFn, primal: &array, cotangent: &array) -> [UniquePtr<array>; 2];

        #[namespace = "mlx_cxx"]
        #[rust_name = "vjp_multiary_fn"]
        unsafe fn vjp(f: *const MultiaryFn, primal: &[UniquePtr<array>], cotangent: &[UniquePtr<array>]) -> [UniquePtr<CxxVector<array>>; 2];

        #[namespace = "mlx_cxx"]
        #[rust_name = "jvp_unary_fn"]
        unsafe fn jvp(f: *const UnaryFn, primal: &array, tangent: &array) -> [UniquePtr<array>; 2];

        #[namespace = "mlx_cxx"]
        #[rust_name = "jvp_multiary_fn"]
        unsafe fn jvp(f: *const MultiaryFn, primal: &[UniquePtr<array>], tangent: &[UniquePtr<array>]) -> [UniquePtr<CxxVector<array>>; 2];
    }
}

#[cfg(test)]
mod tests {
    use crate::cxx_vec;

    #[test]
    fn test_accept_rust_unary_fn() {
        let f = |x: &crate::array::ffi::array| -> cxx::UniquePtr<crate::array::ffi::array> {
            crate::array::ffi::array_new_bool(true)
        };
        let f = super::UnaryFn::from(f);
        let o = super::ffi::accept_rust_unary_fn(&f);
        println!("{}", o);
    }

    #[test]
    fn test_vjp_unary_fn() {
        use std::sync::Arc;

        let shape = cxx_vec!(3i32);
        let b = Arc::new(crate::array::ffi::array_from_slice_float32(&[1.0, 1.0, 1.0], &shape));
        let f = move |arr: &crate::array::ffi::array| -> cxx::UniquePtr<crate::array::ffi::array> {
            crate::ops::ffi::multiply(arr, &**b, Default::default())
        };
        let f = super::UnaryFn::from(f);

        let primal = crate::array::ffi::array_from_slice_float32(&[1.0, 1.0, 1.0], &shape);
        let cotangent = crate::array::ffi::array_from_slice_float32(&[1.0, 1.0, 1.0], &shape);
        unsafe {
            let f_ptr: *const crate::function::UnaryFn = &f;
            let [p, c] = super::ffi::vjp_unary_fn(f_ptr, &primal, &cotangent);
        }
    }
}
