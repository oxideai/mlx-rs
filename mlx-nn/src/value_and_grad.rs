use mlx_rs::{error::Exception, Array};

use crate::{Module, ModuleParameters};

// pub fn value_and_grad<'a, M, F>(model: M, f: F) -> impl FnMut(&M, &Array, &Array) -> (Array, ModuleParameters)
// where
//     M: Module + 'a,
//     F: FnMut(&M, &Array, &Array) -> Array,
// {
//     // let vg = mlx_rs::transforms::value_and_grad(inner, argument_numbers)

//     |model, x, y| {
//         todo!()
//     }
// }

pub trait ValueAndGrad<'a, Args, Output> {
    fn value_and_grad(self) -> impl FnMut(Args) -> Result<(Output, ModuleParameters), Exception>;
}

impl<'a, M, F> ValueAndGrad<'a, (&'a M, &'a Array, &'a Array), Array> for (M, F)
where
    M: Module + 'a,
    F: FnMut(&M, &Array, &Array) -> Array,
{
    fn value_and_grad(
        self,
    ) -> impl FnMut((&'a M, &'a Array, &'a Array)) -> Result<(Array, ModuleParameters), Exception>
    {
        let (model, f) = self;

        move |(model, x, y)| todo!()
    }
}

pub fn value_and_grad<'a, M, F, Args, Output>(
    model: M,
    f: F,
) -> impl FnMut(Args) -> Result<(Output, ModuleParameters), Exception> + 'a
where
    (M, F): ValueAndGrad<'a, Args, Output> + 'a,
    Args: 'a,
    Output: 'a,
{
    (model, f).value_and_grad()
}