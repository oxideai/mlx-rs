use crate::{
    error::{Exception, Result},
    utils::{guard::Guarded, Closure, IntoOption},
    Array,
};

use super::{value_and_gradient, ClosureValueAndGrad};

#[inline]
fn build_gradient_inner<'a>(
    closure: Closure<'a>,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'a {
    move |arrays: &[Array]| -> Result<Vec<Array>> {
        let cvg = ClosureValueAndGrad::try_from_op(|res| unsafe {
            mlx_sys::mlx_value_and_grad(
                res,
                closure.as_ptr(),
                argument_numbers.as_ptr(),
                argument_numbers.len(),
            )
        })?;
        let result = value_and_gradient(cvg.as_ptr(), arrays.iter())?;
        Ok(result.1)
    }
}

fn build_gradient<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'a
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
    let closure = Closure::new(f);
    build_gradient_inner(closure, argument_numbers)
}

fn build_fallible_gradient<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'a
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new_fallible(f);
    build_gradient_inner(closure, argument_numbers)
}

/// Trait for functions/closures that can be converted into a closure that computes the gradient.
pub trait IntoGrad<'a, Args, Output, Err> {
    /// Convert the function/closure into a closure that computes the gradient.
    fn into_grad(
        self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(Args) -> Result<Output> + 'a;
}

impl<'a, F> IntoGrad<'a, &[Array], Vec<Array>, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    // refining_impl_trait is fine here because we have restricted the Args and Output types
    // in the generics.
    #[allow(refining_impl_trait)]
    fn into_grad(
        self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'a {
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        build_gradient(self, argument_numbers)
    }
}

impl<'a, F> IntoGrad<'a, &[Array], Vec<Array>, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'a {
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        build_fallible_gradient(self, argument_numbers)
    }
}

impl<'a, F> IntoGrad<'a, &Array, Array, ()> for F
where
    F: FnMut(&Array) -> Array + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&Array) -> Result<Array> + 'a {
        let f = move |args: &[Array]| -> Vec<Array> { vec![self(&args[0])] };
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        let mut g = build_gradient(f, argument_numbers);
        move |args: &Array| -> Result<Array> {
            let args_clone = &[args.clone()];
            let result = g(args_clone)?;
            Ok(result.into_iter().next().unwrap())
        }
    }
}

impl<'a, F> IntoGrad<'a, &Array, Array, Exception> for F
where
    F: FnMut(&Array) -> Result<Array> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&Array) -> Result<Array> + 'a {
        let f = move |args: &[Array]| -> Result<Vec<Array>> { self(&args[0]).map(|res| vec![res]) };
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        let mut g = build_fallible_gradient(f, argument_numbers);
        move |args: &Array| -> Result<Array> {
            let args_clone = &[args.clone()];
            let result = g(args_clone)?;
            Ok(result.into_iter().next().unwrap())
        }
    }
}

impl<'a, F> IntoGrad<'a, &[Array], Array, ()> for F
where
    F: FnMut(&[Array]) -> Array + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<Array> + 'a {
        let f = move |args: &[Array]| -> Vec<Array> { vec![self(args)] };
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        let mut g = build_gradient(f, argument_numbers);
        move |args: &[Array]| -> Result<Array> {
            let result = g(args)?;
            Ok(result.into_iter().next().unwrap())
        }
    }
}

impl<'a, F> IntoGrad<'a, &[Array], Array, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Array> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<Array> + 'a {
        let f = move |args: &[Array]| -> Result<Vec<Array>> { self(args).map(|res| vec![res]) };
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        let mut g = build_fallible_gradient(f, argument_numbers);
        move |args: &[Array]| -> Result<Array> {
            let result = g(args)?;
            Ok(result.into_iter().next().unwrap())
        }
    }
}

impl<'a, F> IntoGrad<'a, &Array, Vec<Array>, ()> for F
where
    F: FnMut(&Array) -> Vec<Array> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&Array) -> Result<Vec<Array>> + 'a {
        let f = move |args: &[Array]| -> Vec<Array> { self(&args[0]) };
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        let mut g = build_gradient(f, argument_numbers);
        move |args: &Array| -> Result<Vec<Array>> {
            let args_clone = &[args.clone()];
            let result = g(args_clone)?;
            Ok(result)
        }
    }
}

impl<'a, F> IntoGrad<'a, &Array, Vec<Array>, Exception> for F
where
    F: FnMut(&Array) -> Result<Vec<Array>> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&Array) -> Result<Vec<Array>> + 'a {
        let f = move |args: &[Array]| -> Result<Vec<Array>> { self(&args[0]) };
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        let mut g = build_fallible_gradient(f, argument_numbers);
        move |args: &Array| -> Result<Vec<Array>> {
            let args_clone = &[args.clone()];
            let result = g(args_clone)?;
            Ok(result)
        }
    }
}

/// Returns a function which computes the gradient of `f`.
pub fn grad<'a, F, Args, Output, Err>(
    f: F,
    argument_numbers: impl IntoOption<&'a [i32]>,
) -> impl FnMut(Args) -> Result<Output> + 'a
where
    F: IntoGrad<'a, Args, Output, Err>,
{
    f.into_grad(argument_numbers)
}

#[cfg(test)]
mod tests {

    use crate::{
        transforms::{grad, value_and_grad},
        Array,
    };

    // The unit tests below are adapted from the mlx c++ codebase
    #[test]
    fn test_grad() {
        let x = &[Array::from_f32(1.0)];
        let fun = |argin: &[Array]| -> Vec<Array> { vec![&argin[0] + 1.0] };
        let argnums = &[0];

        // TODO: how to make this more "functional"?
        let grad_fn = move |args: &[Array]| -> Vec<Array> { grad(fun, argnums)(args).unwrap() };
        let (z, d2fdx2) = value_and_grad(grad_fn, argnums)(x).unwrap();

        assert_eq!(z[0].item::<f32>(), 1.0);
        assert_eq!(d2fdx2[0].item::<f32>(), 0.0);
    }
}
