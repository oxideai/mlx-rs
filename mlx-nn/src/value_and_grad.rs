use mlx_rs::{error::Exception, Array};

use crate::{update_flattened_parameters, FlattenedModuleParameters, FlattenedModuleParametersRef, Module};

/// Transform the passed function `f(model, args)` to a function that computes the gradients of `f`
/// with regard to the model's trainable parameters and also its value.
/// 
/// TODO: a better name? swift binding uses just `value_and_grad` but the base crate `mlx-rs` also
/// has one
pub fn value_and_grad<'a, M, F, Args>(
    model: &'a mut M,
    mut f: F,
) -> impl FnMut((&'a mut M, Args)) -> Result<(Vec<Array>, FlattenedModuleParameters), Exception> + 'a
where 
    M: Module + 'a,
    F: FnMut(&mut M, Args) -> Vec<Array> + 'a,
    Args: Clone,
{
    let inner = move |(parameters, arrays): (FlattenedModuleParametersRef, Args)| -> Vec<Array> {
        // We either have to clone here or clone inside value_and_grad
        let flattened_parameters = parameters.into_iter().map(|(k, v)| (k, v.clone()));

        // Not sure why the swift binding does this. It seems to be the same parameters
        update_flattened_parameters(model, flattened_parameters);
        f(model, arrays)
    };

    let mut vg = mlx_rs::transforms::value_and_grad_with_hashmap(inner);

    move |(model, arrays)| {
        let trainable_parameters = model.trainable_parameters().flatten();
        let (v, g) = vg((trainable_parameters, arrays))?;
        Ok((v, g))
    }
}