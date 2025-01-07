
use std::{collections::HashMap, rc::Rc};

use mlx_sys::mlx_closure_value_and_grad;

use crate::{
    error::{get_and_clear_closure_error, Exception, Result},
    utils::{guard::Guarded, Closure, IntoOption, VectorArray},
    Array,
};

pub(crate) struct ClosureValueAndGrad {
    pub(crate) c_closure_value_and_grad: mlx_closure_value_and_grad,
}

impl ClosureValueAndGrad {
    pub fn as_ptr(&self) -> mlx_closure_value_and_grad {
        self.c_closure_value_and_grad
    }
}

fn value_and_gradient(
    value_and_grad: mlx_closure_value_and_grad,
    arrays: impl Iterator<Item = impl AsRef<Array>>,
) -> Result<(Vec<Array>, Vec<Array>)> {
    let input_vector = VectorArray::try_from_iter(arrays)?;

    <(Vec<Array>, Vec<Array>) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        mlx_sys::mlx_closure_value_and_grad_apply(
            res_0,
            res_1,
            value_and_grad,
            input_vector.as_ptr(),
        )
    })
    .map_err(|e| match get_and_clear_closure_error() {
        Some(err) => err,
        None => e,
    })
}

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
/// Type alias for a hashmap of parameters.
pub type KeyedParameters<Arr> = HashMap<Rc<str>, Arr>;

/// Type alias for a hashmap of gradients.
pub type KeyedGrad = KeyedParameters<Array>;

macro_rules! keyed_value_and_grad {
    ($inner_ret:ty, $cls_new:ident, $f:ident, $args_ty:ty) => {
        move |parameters: KeyedParameters<Arr>,
              arrays: $args_ty|
              -> Result<(Vec<Array>, KeyedGrad)> {
            let (flattened_keys, flattened_values): (Vec<_>, Vec<_>) =
                parameters.into_iter().unzip();

            let inner = |flattened_arrays: &[Array]| -> $inner_ret {
                let parameters = flattened_keys
                    .iter()
                    .cloned()
                    .zip(flattened_arrays.iter().cloned())
                    .collect();
                ($f)(parameters, arrays.clone())
            };

            let argument_numbers = (0..flattened_values.len() as i32).collect::<Vec<_>>();

            let closure = Closure::$cls_new(inner);
            let cvg = ClosureValueAndGrad::try_from_op(|res| unsafe {
                mlx_sys::mlx_value_and_grad(
                    res,
                    closure.as_ptr(),
                    argument_numbers.as_ptr(),
                    argument_numbers.len(),
                )
            })?;

            let (value, grads) = value_and_gradient(cvg.as_ptr(), flattened_values.into_iter())?;

            let grads_map = flattened_keys.iter().cloned().zip(grads).collect();

            Ok((value, grads_map))
        }
    };
}

/// Similar to [`IntoValueAndGrad`] but for functions that take a hashmap of parameters.
pub trait IntoKeyedValueAndGrad<'a, Arr, Args, Err>
where
    Arr: AsRef<Array>,
    Args: Clone,
{
    /// Convert the function/closure into a closure that computes the value and gradient.
    fn into_keyed_value_and_grad(
        self,
    ) -> impl FnMut(KeyedParameters<Arr>, Args) -> Result<(Vec<Array>, KeyedGrad)> + 'a;
}

impl<'a, F, Arr, Args> IntoKeyedValueAndGrad<'a, Arr, Args, ()> for F
where
    F: FnMut(HashMap<Rc<str>, Array>, Args) -> Vec<Array> + 'a,
    Arr: AsRef<Array>,
    Args: Clone,
{
    fn into_keyed_value_and_grad(
        mut self,
    ) -> impl FnMut(KeyedParameters<Arr>, Args) -> Result<(Vec<Array>, KeyedGrad)> + 'a {
        keyed_value_and_grad!(Vec<Array>, new, self, Args)
    }
}

impl<'a, F, Arr, Args> IntoKeyedValueAndGrad<'a, Arr, Args, Exception> for F
where
    F: FnMut(HashMap<Rc<str>, Array>, Args) -> Result<Vec<Array>> + 'a,
    Arr: AsRef<Array>,
    Args: Clone,
{
    fn into_keyed_value_and_grad(
        mut self,
    ) -> impl FnMut(KeyedParameters<Arr>, Args) -> Result<(Vec<Array>, KeyedGrad)> + 'a {
        keyed_value_and_grad!(Result<Vec<Array>>, new_fallible, self, Args)
    }
}

/// Returns a function which computes the value and gradient of `f` with keyed parameters.
pub fn keyed_value_and_grad<'a, F, Arr, Args, Err>(
    f: F,
) -> impl FnMut(KeyedParameters<Arr>, Args) -> Result<(Vec<Array>, KeyedGrad)> + 'a
where
    F: IntoKeyedValueAndGrad<'a, Arr, Args, Err> + 'a,
    Arr: AsRef<Array>,
    Args: Clone,
{
    f.into_keyed_value_and_grad()
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
    use std::{collections::HashMap, rc::Rc};

    use crate::{
        array,
        transforms::{grad, jvp, value_and_grad, vjp},
        Array,
    };

    use super::*;

    use super::keyed_value_and_grad;

    // The unit tests below are adapted from the mlx c++ codebase
    #[test]
    fn test_value_and_grad() {
        let x = &[Array::from_float(1.0)];
        let fun = |argin: &[Array]| -> Vec<Array> { vec![&argin[0] + 1.0] };
        let argnums = &[0];
        let (y, dfdx) = value_and_grad(fun, argnums)(x).unwrap();

        assert_eq!(y[0].item::<f32>(), 2.0);
        assert_eq!(dfdx[0].item::<f32>(), 1.0);

        // TODO: how to make this more "functional"?
        let grad_fn = move |args: &[Array]| -> Vec<Array> { grad(fun, argnums)(args).unwrap() };
        let (z, d2fdx2) = value_and_grad(grad_fn, argnums)(x).unwrap();

        assert_eq!(z[0].item::<f32>(), 1.0);
        assert_eq!(d2fdx2[0].item::<f32>(), 0.0);
    }

    #[test]
    fn test_value_and_grad_hash_map() {
        let f = |parameters: HashMap<Rc<str>, Array>, _: i32| -> Vec<Array> {
            vec![&parameters["x"] * &parameters["y"]]
        };

        let x = array!(1.5f32);
        let y = array!(2.0f32);
        let parameters = vec![("x", x), ("y", y)]
            .into_iter()
            .map(|(k, v)| (k.into(), v))
            .collect();

        let mut vg = keyed_value_and_grad(f);

        let (value, grad) = vg(parameters, 0).unwrap();

        assert_eq!(value[0].item::<f32>(), 1.5 * 2.0);
        assert_eq!(grad["x"].item::<f32>(), 2.0);
        assert_eq!(grad["y"].item::<f32>(), 1.5);
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
