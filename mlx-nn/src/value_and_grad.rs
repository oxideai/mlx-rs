use mlx_rs::{error::Exception, Array};

use crate::module::{FlattenedModuleParam, FlattenedModuleParamRef, Module};

/// Transform the passed function `f(model, args)` to a function that computes the gradients of `f`
/// with regard to the model's trainable parameters and also its value.
///
/// TODO: a better name? swift binding uses just `value_and_grad` but the base crate `mlx-rs` also
/// has one
pub fn value_and_grad<'a, M, F, Args>(
    model: &'a M,
    mut f: F,
) -> impl FnMut(Args) -> Result<(Vec<Array>, FlattenedModuleParam), Exception> + 'a
where
    M: Module + 'a,
    F: FnMut(&M, Args) -> Vec<Array> + 'a,
    Args: Clone,
{
    // We need to have the parameters in the closure so that the gradient will be computed
    // wrt them
    let inner = move |(_parameters, arrays): (FlattenedModuleParamRef, Args)| -> Vec<Array> {
        // We either have to clone here or clone inside value_and_grad
        // let flattened_parameters = parameters.into_iter().map(|(k, v)| (k, v.clone()));

        // This is needed in the swift binding, otherwise the gradients will all be zero
        // However, we might not need this
        // update_flattened_parameters(model, flattened_parameters);
        f(model, arrays)
    };

    let mut vg = mlx_rs::transforms::value_and_grad_with_hashmap(inner);

    move |arrays| {
        let trainable_parameters = model.trainable_parameters().flatten();
        let (v, g) = vg((trainable_parameters, arrays))?;
        Ok((v, g))
    }
}

#[cfg(test)]
mod tests {}
