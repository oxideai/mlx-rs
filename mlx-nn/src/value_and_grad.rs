use mlx_nn_module::update_flattened_parameters;
use mlx_rs::{error::Exception, Array};

use crate::module::{FlattenedModuleParam, FlattenedModuleParamRef, Module};

/// Helper trait for [`value_and_grad`]
pub trait IntoModuleValueAndGrad<'a, M, Args, Err>
where
    M: Module + 'a,
    Args: Clone,
{
    /// Computes the valud and gradient of the passed function `f(model, args)` with regard to the
    /// model's trainable parameters.
    fn into_module_value_and_grad(
        self,
    ) -> impl FnMut(&'a mut M, Args) -> Result<(Vec<Array>, FlattenedModuleParam), Exception> + 'a;
}

macro_rules! into_module_value_and_grad_inner {
    ($inner_ret_ty:ty, $f:ident) => {
        move |model, arrays| {
            // We have to clone here to avoid issue with the mutable borrow of `model` in the closure
            let trainable_parameters = model
                .trainable_parameters()
                .flatten()
                .into_iter()
                .map(|(k, v)| (k, v.clone()))
                .collect();

            // We need to have the parameters in the closure so that the gradient will be computed wrt
            // them
            let inner = |parameters: FlattenedModuleParamRef, arrays: Args| -> $inner_ret_ty {
                // Somehow the parameters of the model captured in the closure are not the same arrays
                // as the ones passed in the outer function (their memory address are actually different
                // in the swift binding).
                //
                // We need to update the parameters of the model with the ones passed in, otherwise the
                // gradients will be zero
                let flattened_parameters = parameters.into_iter().map(|(k, v)| (k, v.clone()));
                update_flattened_parameters(model, flattened_parameters);

                ($f)(model, arrays)
            };
            let mut vg = mlx_rs::transforms::value_and_grad_with_hashmap(inner);

            let (v, g) = vg(trainable_parameters, arrays)?;
            Ok((v, g))
        }
    };
}

impl<'a, F, M, Args> IntoModuleValueAndGrad<'a, M, Args, ()> for F
where
    M: Module + 'a,
    F: FnMut(&M, Args) -> Vec<Array> + 'a,
    Args: Clone,
{
    fn into_module_value_and_grad(
        mut self,
    ) -> impl FnMut(&'a mut M, Args) -> Result<(Vec<Array>, FlattenedModuleParam), Exception> + 'a
    {
        into_module_value_and_grad_inner!(Vec<Array>, self)
    }
}

impl<'a, F, M, Args> IntoModuleValueAndGrad<'a, M, Args, Exception> for F
where
    M: Module + 'a,
    F: FnMut(&M, Args) -> Result<Vec<Array>, Exception> + 'a,
    Args: Clone,
{
    fn into_module_value_and_grad(
        mut self,
    ) -> impl FnMut(&'a mut M, Args) -> Result<(Vec<Array>, FlattenedModuleParam), Exception> + 'a
    {
        into_module_value_and_grad_inner!(Result<Vec<Array>, Exception>, self)
    }
}

/// Transform the passed function `f(model, args)` to a function that computes the gradients of `f`
/// with regard to the model's trainable parameters and also its value.
pub fn module_value_and_grad<'a, F, M, Args, Err>(
    f: F,
) -> impl FnMut(&'a mut M, Args) -> Result<(Vec<Array>, FlattenedModuleParam), Exception> + 'a
where
    M: Module + 'a,
    F: IntoModuleValueAndGrad<'a, M, Args, Err>,
    Args: Clone,
{
    f.into_module_value_and_grad()
}

#[cfg(test)]
mod tests {
    use mlx_nn_module::Module;
    use mlx_rs::{array, error::Exception, Array};

    use crate::Linear;

    use super::*;

    // The unit test below is adapted from `test_compiled_optimizer` in
    // `mlx/python/tests/test_optimizers.py``
    #[test]
    fn test_module_value_and_grad() {
        let mut model = Linear::new(2, 2).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(1.0, 2.0, &[2, 2], None).unwrap();

        let loss = |model: &Linear, x: &Array| -> Vec<Array> {
            vec![model.forward(x).unwrap().sum(None, None).unwrap()]
        };

        let mut vg = module_value_and_grad(loss);
        let (v, g) = vg(&mut model, &x).unwrap();

        assert_ne!(v[0].sum(None, None).unwrap(), array!(0.0));
        assert_ne!(g["weight"].sum(None, None).unwrap(), array!(0.0));
        assert_ne!(g["bias"].sum(None, None).unwrap(), array!(0.0));
    }

    #[test]
    fn test_fallible_module_value_and_grad() {
        let mut model = Linear::new(2, 2).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(1.0, 2.0, &[2, 2], None).unwrap();

        let loss = |model: &Linear, x: &Array| -> Result<Vec<Array>, Exception> {
            Ok(vec![model.forward(x)?.sum(None, None)?])
        };

        let mut vg = module_value_and_grad(loss);
        let (v, g) = vg(&mut model, &x).unwrap();

        assert_ne!(v[0].sum(None, None).unwrap(), array!(0.0));
        assert_ne!(g["weight"].sum(None, None).unwrap(), array!(0.0));
        assert_ne!(g["bias"].sum(None, None).unwrap(), array!(0.0));
    }
}
