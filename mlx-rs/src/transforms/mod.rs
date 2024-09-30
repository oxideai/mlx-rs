use std::{collections::HashMap, rc::Rc};

use mlx_sys::{mlx_closure_value_and_grad, mlx_closure_value_and_grad_apply};
use smallvec::SmallVec;

use crate::{
    error::{
        get_and_clear_last_mlx_error, is_mlx_error_handler_set, setup_mlx_error_handler, Exception,
    },
    utils::{Closure, IntoOption, VectorArray, VectorVectorArray},
    Array,
};

pub mod compile;

/// Evaluate an iterator of [`Array`]s.
pub fn eval<'a>(outputs: impl IntoIterator<Item = &'a mut Array>) -> Result<(), Exception> {
    if !is_mlx_error_handler_set() {
        setup_mlx_error_handler();
    }

    let vec = VectorArray::from_iter(outputs.into_iter());

    unsafe {
        mlx_sys::mlx_eval(vec.as_ptr());
    }

    get_and_clear_last_mlx_error().map_or(Ok(()), Err)
}

/// Asynchronously evaluate an iterator of [`Array`]s.
///
/// Please note that this is not a rust async function.
pub fn async_eval<'a>(outputs: impl IntoIterator<Item = &'a mut Array>) -> Result<(), Exception> {
    if !is_mlx_error_handler_set() {
        setup_mlx_error_handler();
    }

    let vec = VectorArray::from_iter(outputs.into_iter());

    unsafe {
        mlx_sys::mlx_async_eval(vec.as_ptr());
    }

    get_and_clear_last_mlx_error().map_or(Ok(()), Err)
}

#[inline]
fn jvp_inner(
    closure: Closure<'_>,
    primals: &[Array],
    tangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>), Exception> {
    let c_primals = VectorArray::from_iter(primals.iter());
    let c_tangents = VectorArray::from_iter(tangents.iter());

    let vector_pair = unsafe {
        let c_vector_pair = try_catch_mlx_closure_error! {
            mlx_sys::mlx_jvp(
                closure.as_ptr(),
                c_primals.as_ptr(),
                c_tangents.as_ptr(),
            )
        };
        VectorVectorArray::from_ptr(c_vector_pair)
    };

    let vector_pair_values: SmallVec<[VectorArray; 2]> = vector_pair.into_values();
    let mut iter = vector_pair_values.into_iter();
    let v1 = iter.next().unwrap().into_values();
    let v2 = iter.next().unwrap().into_values();

    Ok((v1, v2))
}

/// Compute the Jacobian-vector product.
///
/// This computes the product of the Jacobian of a function `f` evaluated at `primals` with the
/// `tangents`.
///
/// # Params:
///
/// - `f`: function which takes an array of `Array` and returns an array of `Array`
/// - `primals`: array of `Array` at which to evaluate the Jacobian
/// - `tangents`: array of `Array` which are the "vector" in the Jacobian-vector product.  The
///     `tangents` should be the same in number, shape and type as the inputs of `f`, e.g. the
///     `primals`
///
/// # Returns:
///
/// Array of the Jacobian-vector products which is the same in number, shape and type of
/// the outputs of `f`
pub fn jvp<'a, F>(
    f: F,
    primals: &[Array],
    tangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>), Exception>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);
    jvp_inner(closure, primals, tangents)
}

/// Similar to [`jvp`] but handles closures that can return an error.
pub fn jvp_fallible<'a, F>(
    f: F,
    primals: &[Array],
    tangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>), Exception>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    let closure = Closure::new_fallible(f);
    jvp_inner(closure, primals, tangents)
}

#[inline]
fn vjp_inner(
    closure: Closure<'_>,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>), Exception> {
    let c_primals = VectorArray::from_iter(primals.iter());
    let c_cotangents = VectorArray::from_iter(cotangents.iter());

    let vector_pair = unsafe {
        let c_vector_pair = try_catch_mlx_closure_error! {
            mlx_sys::mlx_vjp(
                closure.as_ptr(),
                c_primals.as_ptr(),
                c_cotangents.as_ptr(),
            )
        };
        VectorVectorArray::from_ptr(c_vector_pair)
    };

    let vector_pair_values: SmallVec<[VectorArray; 2]> = vector_pair.into_values();
    let mut iter = vector_pair_values.into_iter();
    let v1 = iter.next().unwrap().into_values();
    let v2 = iter.next().unwrap().into_values();

    Ok((v1, v2))
}

/// Compute the vector-Jacobian product.
///
/// Computes the product of the `cotangents` with the Jacobian of a function `f` evaluated at
/// `primals`.
///
/// # Params:
///
/// - f: function which takes an array of `Array` and returns an array of `Array`
/// - primals: array of `Array` at which to evaluate the Jacobian
/// - cotangents: array of `Array` which are the "vector" in the vector-Jacobian product. The
///   `cotangents` should be the same in number, shape and type as the outputs of `f`
///
/// # Returns:
///
/// array of the vector-Jacobian products which is the same in number, shape and type of the outputs
/// of `f`
pub fn vjp<'a, F>(
    f: F,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>), Exception>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);
    vjp_inner(closure, primals, cotangents)
}

/// Similar to [`vjp`] but handles closures that can return an error.
pub fn vjp_fallible<'a, F>(
    f: F,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>), Exception>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    let closure = Closure::new_fallible(f);
    vjp_inner(closure, primals, cotangents)
}

fn value_and_gradient(
    value_and_grad: mlx_closure_value_and_grad,
    arrays: impl Iterator<Item = impl AsRef<Array>>,
) -> Result<(Vec<Array>, Vec<Array>), Exception> {
    let input_vector = VectorArray::from_iter(arrays);

    let vector_pair = unsafe {
        let c_vector_pair = try_catch_mlx_closure_error! {
            mlx_closure_value_and_grad_apply(value_and_grad, input_vector.as_ptr())
        };
        VectorVectorArray::from_ptr(c_vector_pair)
    };

    let vector_pair_values: SmallVec<[VectorArray; 2]> = vector_pair.into_values();
    let mut iter = vector_pair_values.into_iter();
    let values_vec = iter.next().unwrap().into_values();
    let gradients_vec = iter.next().unwrap().into_values();

    Ok((values_vec, gradients_vec))
}

#[inline]
fn build_gradient_inner<'a>(
    closure: Closure<'a>,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a {
    move |arrays: &[Array]| -> Result<Vec<Array>, Exception> {
        unsafe {
            let c_value_and_grad = try_catch_mlx_closure_error! {
                mlx_sys::mlx_value_and_grad(
                    closure.as_ptr(),
                    argument_numbers.as_ptr(),
                    argument_numbers.len(),
                )
            };

            let result = value_and_gradient(c_value_and_grad, arrays.iter())?;
            Ok(result.1)
        }
    }
}

fn build_gradient<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
    let closure = Closure::new(f);
    build_gradient_inner(closure, argument_numbers)
}

fn build_gradient_fallible<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    let closure = Closure::new_fallible(f);
    build_gradient_inner(closure, argument_numbers)
}

fn build_value_and_gradient_inner<'a>(
    closure: Closure<'a>,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>), Exception> + 'a {
    move |arrays: &[Array]| {
        let c_value_and_grad = unsafe {
            try_catch_mlx_closure_error! {
                mlx_sys::mlx_value_and_grad(
                    closure.as_ptr(),
                    argument_numbers.as_ptr(),
                    argument_numbers.len(),
                )
            }
        };

        value_and_gradient(c_value_and_grad, arrays.iter())
    }
}

fn build_value_and_gradient<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>), Exception> + 'a
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);
    build_value_and_gradient_inner(closure, argument_numbers)
}

fn build_value_and_gradient_fallible<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>), Exception> + 'a
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    let closure = Closure::new_fallible(f);
    build_value_and_gradient_inner(closure, argument_numbers)
}

/// Returns a function which computes the value and gradient of `f`.
pub fn value_and_grad<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>), Exception> + 'a
{
    build_value_and_gradient(f, argument_numbers)
}

pub type HashMapGrad = HashMap<Rc<str>, Array>;

pub fn value_and_grad_with_hashmap<'a, F, T, Arr>(
    mut f: F,
) -> impl FnMut(HashMap<Rc<str>, Arr>, T) -> Result<(Vec<Array>, HashMapGrad), Exception> + 'a
where
    F: FnMut(HashMap<Rc<str>, &Array>, T) -> Vec<Array> + 'a,
    Arr: AsRef<Array>,
    T: Clone,
{
    move |parameters: HashMap<Rc<str>, Arr>,
          arrays: T|
          -> Result<(Vec<Array>, HashMapGrad), Exception> {
        let (flattened_keys, flattened_values): (Vec<_>, Vec<_>) = parameters.into_iter().unzip();

        let inner = |flattened_arrays: &[Array]| -> Vec<Array> {
            let parameters = flattened_keys
                .iter()
                .cloned()
                .zip(flattened_arrays)
                .collect();
            f(parameters, arrays.clone())
        };

        let argument_numbers = (0..flattened_values.len() as i32).collect::<Vec<_>>();

        let closure = Closure::new(inner);
        let c_value_and_grad = unsafe {
            try_catch_c_ptr_expr! {
                mlx_sys::mlx_value_and_grad(
                    closure.as_ptr(),
                    argument_numbers.as_ptr(),
                    argument_numbers.len(),
                )
            }
        };

        let (value, grads) = value_and_gradient(c_value_and_grad, flattened_values.into_iter())?;

        let grads_map = flattened_keys.iter().cloned().zip(grads).collect();

        Ok((value, grads_map))
    }
}

pub fn value_and_grad_fallible<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>), Exception> + 'a
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    build_value_and_gradient_fallible(f, argument_numbers)
}

pub trait Grad<'a, Args, Output, Err> {
    fn grad(
        self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(Args) -> Result<Output, Exception> + 'a;
}

impl<'a, F> Grad<'a, &[Array], Vec<Array>, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    // refining_impl_trait is fine here because we have restricted the Args and Output types
    // in the generics.
    #[allow(refining_impl_trait)]
    fn grad(
        self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a {
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        build_gradient(self, argument_numbers)
    }
}

impl<'a, F> Grad<'a, &[Array], Vec<Array>, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    #[allow(refining_impl_trait)]
    fn grad(
        self,
        argument_numbers: &'a [i32],
    ) -> impl FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a {
        build_gradient_fallible(self, argument_numbers)
    }
}

impl<'a, F> Grad<'a, &Array, Array, ()> for F
where
    F: FnMut(&Array) -> Array + 'a,
{
    #[allow(refining_impl_trait)]
    fn grad(
        mut self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&Array) -> Result<Array, Exception> + 'a {
        let f = move |args: &[Array]| -> Vec<Array> { vec![self(&args[0])] };
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        let mut g = build_gradient(f, argument_numbers);
        move |args: &Array| -> Result<Array, Exception> {
            let args_clone = &[args.clone()];
            let result = g(args_clone)?;
            Ok(result.into_iter().next().unwrap())
        }
    }
}

impl<'a, F> Grad<'a, &Array, Array, Exception> for F
where
    F: FnMut(&Array) -> Result<Array, Exception> + 'a,
{
    #[allow(refining_impl_trait)]
    fn grad(
        mut self,
        argument_numbers: &'a [i32],
    ) -> impl FnMut(&Array) -> Result<Array, Exception> + 'a {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            self(&args[0]).map(|res| vec![res])
        };
        let mut g = build_gradient_fallible(f, argument_numbers);
        move |args: &Array| -> Result<Array, Exception> {
            let args_clone = &[args.clone()];
            let result = g(args_clone)?;
            Ok(result.into_iter().next().unwrap())
        }
    }
}

impl<'a, F> Grad<'a, &[Array], Array, ()> for F
where
    F: FnMut(&[Array]) -> Array + 'a,
{
    #[allow(refining_impl_trait)]
    fn grad(
        mut self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<Array, Exception> + 'a {
        let f = move |args: &[Array]| -> Vec<Array> { vec![self(args)] };
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        let mut g = build_gradient(f, argument_numbers);
        move |args: &[Array]| -> Result<Array, Exception> {
            let result = g(args)?;
            Ok(result.into_iter().next().unwrap())
        }
    }
}

impl<'a, F> Grad<'a, &[Array], Array, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Array, Exception> + 'a,
{
    #[allow(refining_impl_trait)]
    fn grad(
        mut self,
        argument_numbers: &'a [i32],
    ) -> impl FnMut(&[Array]) -> Result<Array, Exception> + 'a {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            self(args).map(|res| vec![res])
        };
        let mut g = build_gradient_fallible(f, argument_numbers);
        move |args: &[Array]| -> Result<Array, Exception> {
            let result = g(args)?;
            Ok(result.into_iter().next().unwrap())
        }
    }
}

impl<'a, F> Grad<'a, &Array, Vec<Array>, ()> for F
where
    F: FnMut(&Array) -> Vec<Array> + 'a,
{
    #[allow(refining_impl_trait)]
    fn grad(
        mut self,
        argument_numbers: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&Array) -> Result<Vec<Array>, Exception> + 'a {
        let f = move |args: &[Array]| -> Vec<Array> { self(&args[0]) };
        let argument_numbers = argument_numbers.into_option().unwrap_or(&[0]);
        let mut g = build_gradient(f, argument_numbers);
        move |args: &Array| -> Result<Vec<Array>, Exception> {
            let args_clone = &[args.clone()];
            let result = g(args_clone)?;
            Ok(result)
        }
    }
}

impl<'a, F> Grad<'a, &Array, Vec<Array>, Exception> for F
where
    F: FnMut(&Array) -> Result<Vec<Array>, Exception> + 'a,
{
    #[allow(refining_impl_trait)]
    fn grad(
        mut self,
        argument_numbers: &'a [i32],
    ) -> impl FnMut(&Array) -> Result<Vec<Array>, Exception> + 'a {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> { self(&args[0]) };
        let mut g = build_gradient_fallible(f, argument_numbers);
        move |args: &Array| -> Result<Vec<Array>, Exception> {
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
) -> impl FnMut(Args) -> Result<Output, Exception> + 'a
where
    F: Grad<'a, Args, Output, Err>,
{
    f.grad(argument_numbers)
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

    use super::value_and_grad_with_hashmap;

    // The unit tests below are adapted from the mlx c++ codebase

    #[test]
    fn test_jvp() {
        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] + &inputs[1]] };
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let (out, dout) = jvp(f, &[x, y], &[array!(1.0f32), array!(3.0f32)]).unwrap();
        assert_eq!(out[0].item::<f32>(), 2.0f32);
        assert_eq!(dout[0].item::<f32>(), 4.0f32);
    }

    #[test]
    fn test_jvp_with_error() {
        let f = |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            inputs[0].add(&inputs[1]).map(|res| vec![res])
        };

        // Success case
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let (out, dout) = jvp_fallible(f, &[x, y], &[array!(1.0f32), array!(3.0f32)]).unwrap();
        assert_eq!(out[0].item::<f32>(), 2.0f32);
        assert_eq!(dout[0].item::<f32>(), 4.0f32);

        // Error case
        // Use non-broadcastable shapes
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let result = jvp_fallible(f, &[a, b], &[array!(1.0f32), array!(3.0f32)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vjp() {
        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] + &inputs[1]] };
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let primals = vec![x, y];
        let cotangents = vec![array!(1.0f32)];
        let (out, dout) = vjp(f, &primals, &cotangents).unwrap();
        assert_eq!(out[0].item::<f32>(), 2.0f32);
        assert_eq!(dout[0].item::<f32>(), 1.0f32);
    }

    #[test]
    fn test_vjp_with_error() {
        let f = |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            inputs[0].add(&inputs[1]).map(|res| vec![res])
        };

        // Success case
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let primals = vec![x, y];
        let cotangents = vec![array!(1.0f32)];
        let (out, dout) = vjp_fallible(f, &primals, &cotangents).unwrap();
        assert_eq!(out[0].item::<f32>(), 2.0f32);
        assert_eq!(dout[0].item::<f32>(), 1.0f32);

        // Error case
        // Use non-broadcastable shapes
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let result = vjp_fallible(f, &[a, b], &[array!(1.0f32)]);
        assert!(result.is_err());
    }

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
    fn test_value_and_grad() {
        let f = |x: &[Array]| -> Vec<Array> { vec![&x[0] * &x[0]] };

        let x = array!(1.5f32);

        let mut vg = value_and_grad(f, None);

        let (value, grad) = vg(&[x]).unwrap();

        assert_eq!(value[0].item::<f32>(), 1.5 * 1.5);
        assert_eq!(grad[0].item::<f32>(), 2.0 * 1.5);
    }

    #[test]
    fn test_value_and_grad_hash_map() {
        let f = |parameters: HashMap<Rc<str>, &Array>, _: i32| -> Vec<Array> {
            vec![parameters["x"] * parameters["y"]]
        };

        let x = array!(1.5f32);
        let y = array!(2.0f32);
        let parameters = vec![("x", &x), ("y", &y)]
            .into_iter()
            .map(|(k, v)| (k.into(), v))
            .collect();

        let mut vg = value_and_grad_with_hashmap(f);

        let (value, grad) = vg(parameters, 0).unwrap();

        assert_eq!(value[0].item::<f32>(), 1.5 * 2.0);
        assert_eq!(grad["x"].item::<f32>(), 2.0);
        assert_eq!(grad["y"].item::<f32>(), 1.5);
    }

    fn test_value_and_grad_with_error() {
        let fun = |argin: &[Array]| -> Result<Vec<Array>, Exception> {
            argin[0].add(array!(1.0)).map(|res| vec![res])
        };

        // Success case
        let argnums = &[0];
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let result = value_and_grad_fallible(fun, argnums)(&[x, y]);
        assert!(result.is_ok());

        // Error case
        // Use non-broadcastable shapes
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let result = value_and_grad_fallible(fun, argnums)(&[a, b]);
        assert!(result.is_err());
    }
}
