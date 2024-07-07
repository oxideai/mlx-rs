use mlx_sys::{mlx_closure_value_and_grad, mlx_closure_value_and_grad_apply};
use smallvec::SmallVec;

use crate::{
    error::{
        get_and_clear_last_mlx_error, is_mlx_error_handler_set, setup_mlx_error_handler, Exception,
    },
    utils::{Closure, VectorArray, VectorVectorArray},
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
    F: FnOnce(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);

    let c_primals = VectorArray::from_iter(primals.iter());
    let c_tangents = VectorArray::from_iter(tangents.iter());

    let vector_pair = unsafe {
        let c_vector_pair = try_catch_c_ptr_expr! {
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
    F: FnOnce(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);

    let c_primals = VectorArray::from_iter(primals.iter());
    let c_cotangents = VectorArray::from_iter(cotangents.iter());

    let vector_pair = unsafe {
        let c_vector_pair = try_catch_c_ptr_expr! {
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

fn value_and_gradient(
    value_and_grad: mlx_closure_value_and_grad,
    arrays: impl Iterator<Item = impl AsRef<Array>>,
) -> Result<(Vec<Array>, Vec<Array>), Exception> {
    let input_vector = VectorArray::from_iter(arrays);

    let vector_pair = unsafe {
        // TODO: replace with VectorVectorArray once #26 is merged
        let c_vector_pair = try_catch_c_ptr_expr! {
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

fn build_gradient<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnOnce(&[Array]) -> Result<Vec<Array>, Exception> + 'a
where
    F: FnOnce(&[Array]) -> Vec<Array> + 'a,
{
    move |arrays: &[Array]| -> Result<Vec<Array>, Exception> {
        unsafe {
            let closure = Closure::new(f);
            let c_value_and_grad = try_catch_c_ptr_expr! {
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

fn build_value_and_gradient<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnOnce(&[Array]) -> Result<(Vec<Array>, Vec<Array>), Exception> + 'a
where
    F: FnOnce(&[Array]) -> Vec<Array> + 'a,
{
    |arrays: &[Array]| {
        let closure = Closure::new(f);
        let c_value_and_grad = unsafe {
            try_catch_c_ptr_expr! {
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

/// Returns a function which computes the value and gradient of `f`.
pub fn value_and_grad<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnOnce(&[Array]) -> Result<(Vec<Array>, Vec<Array>), Exception> + 'a
where
    F: FnOnce(&[Array]) -> Vec<Array> + 'a,
{
    build_value_and_gradient(f, argument_numbers)
}

/// Returns a function which computes the gradient of `f`.
pub fn grad<'a, F>(
    f: F,
    argument_numbers: &'a [i32],
) -> impl FnOnce(&[Array]) -> Result<Vec<Array>, Exception> + 'a
where
    F: FnOnce(&[Array]) -> Vec<Array> + 'a,
{
    build_gradient(f, argument_numbers)
}

/// Helper trait to make working with transforms ops more ergonomic.
pub trait TransformsExt: Sized {
    /// The Ok output type of Self.
    type Ok;

    /// Compose the current closure with another closure.
    fn and_then<F, T>(self, op: F) -> impl FnOnce(&[Array]) -> Result<T, Exception>
    where
        F: FnOnce(&[Array]) -> Result<T, Exception>;

    /// Returns a closure that unwraps the result of the current closure.
    fn unwrapped(self) -> impl FnOnce(&[Array]) -> Self::Ok
    where
        Self: FnOnce(&[Array]) -> Result<Self::Ok, Exception>,
    {
        move |args| self(args).unwrap()
    }
}

impl<F> TransformsExt for F
where
    for<'a> F: FnOnce(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    type Ok = Vec<Array>;

    fn and_then<F2, T>(self, op: F2) -> impl FnOnce(&[Array]) -> Result<T, Exception>
    where
        F2: FnOnce(&[Array]) -> Result<T, Exception>,
    {
        move |args| {
            let arrays = self(args)?;
            op(&arrays)
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        array,
        transforms::{grad, jvp, value_and_grad, vjp, TransformsExt},
        Array,
    };

    // The unit tests below are adapted from the mlx c++ codebase

    #[test]
    fn test_jvp() {
        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] + &inputs[1]] };
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let (mut out, mut dout) = jvp(f, &[x, y], &[array!(1.0f32), array!(3.0f32)]).unwrap();
        assert_eq!(out[0].item::<f32>(), 2.0f32);
        assert_eq!(dout[0].item::<f32>(), 4.0f32);
    }

    #[test]
    fn test_vjp() {
        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] + &inputs[1]] };
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let primals = vec![x, y];
        let cotangents = vec![array!(1.0f32)];
        let (mut out, mut dout) = vjp(f, &primals, &cotangents).unwrap();
        assert_eq!(out[0].item::<f32>(), 2.0f32);
        assert_eq!(dout[0].item::<f32>(), 1.0f32);
    }

    #[test]
    fn test_grad() {
        let x = &[Array::from_float(1.0)];
        let fun = |argin: &[Array]| -> Vec<Array> { vec![&argin[0] + 1.0] };
        let argnums = &[0];
        let (mut y, mut dfdx) = value_and_grad(fun, argnums)(x).unwrap();

        assert_eq!(y[0].item::<f32>(), 2.0);
        assert_eq!(dfdx[0].item::<f32>(), 1.0);

        // TODO: how to make this more "functional"?
        let (mut z, mut d2fdx2) =
            value_and_grad(grad(fun, argnums).unwrapped(), argnums)(x).unwrap();

        assert_eq!(z[0].item::<f32>(), 1.0);
        assert_eq!(d2fdx2[0].item::<f32>(), 0.0);
    }
}
