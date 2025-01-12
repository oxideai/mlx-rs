use crate::{
    error::{Exception, Result},
    utils::{guard::Guarded, Closure, IntoOption},
    Array,
};

use super::{value_and_gradient, ClosureValueAndGrad};

fn build_value_and_gradient_inner<'a>(
    closure: Closure<'a>,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a {
    move |arrays: &[Array]| unsafe {
        let cvg = ClosureValueAndGrad::try_from_op(|res| {
            mlx_sys::mlx_value_and_grad(
                res,
                closure.as_ptr(),
                argument_numbers.as_ptr(),
                argument_numbers.len(),
            )
        })?;
        value_and_gradient(cvg.as_ptr(), arrays.iter())
    }
}

fn build_value_and_gradient<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);
    build_value_and_gradient_inner(closure, argument_numbers)
}

fn build_fallible_value_and_gradient<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new_fallible(f);
    build_value_and_gradient_inner(closure, argument_numbers)
}

/// Trait for functions/closures that can be converted into a closure that computes the value and
/// gradient.
pub trait IntoValueAndGrad<'a, Err> {
    /// Convert the function/closure into a closure that computes the value and gradient.
    fn into_value_and_grad(
        self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a;
}

impl<'a, F> IntoValueAndGrad<'a, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    // refining_impl_trait is fine here because we have restricted the Args and Output types
    // in the generics.
    #[allow(refining_impl_trait)]
    fn into_value_and_grad(
        self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a {
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        build_value_and_gradient(self, argument_numbers)
    }
}

impl<'a, F> IntoValueAndGrad<'a, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_value_and_grad(
        self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a {
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        build_fallible_value_and_gradient(self, argument_numbers)
    }
}

/// Returns a function which computes the value and gradient of `f`.
pub fn value_and_grad<'a, F, Err>(
    f: F,
    argument_numbers: impl IntoOption<&'a [i32]>,
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a
where
    F: IntoValueAndGrad<'a, Err> + 'a,
{
    f.into_value_and_grad(argument_numbers)
}

#[cfg(test)]
mod tests {

    use crate::{array, transforms::value_and_grad, Array};

    use super::*;

    // The unit tests below are adapted from the mlx c++ codebase
    #[test]
    fn test_value_and_grad() {
        let x = &[Array::from_float(1.0)];
        let fun = |argin: &[Array]| -> Vec<Array> { vec![&argin[0] + 1.0] };
        let argnums = &[0];
        let (y, dfdx) = value_and_grad(fun, argnums)(x).unwrap();

        assert_eq!(y[0].item::<f32>(), 2.0);
        assert_eq!(dfdx[0].item::<f32>(), 1.0);
    }

    #[test]
    fn test_value_and_grad_with_error() {
        let fun = |argin: &[Array]| -> Result<Vec<Array>> {
            argin[0].add(array!(1.0)).map(|res| vec![res])
        };

        // Success case
        let argnums = &[0];
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let result = value_and_grad(fun, argnums)(&[x, y]);
        assert!(result.is_ok());

        // Error case
        // Use non-broadcastable shapes
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let result = value_and_grad(fun, argnums)(&[a, b]);
        assert!(result.is_err());

        // Check that the error is not just "mlx_closure returned a non-zero value"
        let err = result.unwrap_err();
        assert!(!err.what().contains("non-zero value"))
    }
}
