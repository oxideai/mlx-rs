use std::marker::PhantomData;

use crate::{
    error::Exception,
    Array,
};

use super::{type_id_to_usize, Compiled, CompiledState};

/// A trait for functions that can be compiled.
///
/// # Generics:
/// 
/// - `A`: The type of the array arguments
/// - `O`: The type of the output
/// - `E`: The type of the error
pub trait Compile<A, O, E>: Sized {
    fn compile(
        self,
        shapeless: bool,
    ) -> impl CallMut<A, O, E>;
}

impl<'a, F> Compile<&'a [Array], Vec<Array>, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    fn compile(
        self,
        shapeless: bool,
    ) -> impl CallMut<&'a [Array], Vec<Array>, ()> {
        let id = type_id_to_usize(&self);
        let state = CompiledState {
            f: self,

            shapeless,
            id,
            
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<&'a [Array], Vec<Array>, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    fn compile(
        self,
        shapeless: bool,
    ) -> impl CallMut<&'a [Array], Vec<Array>, Exception> {
        let id = type_id_to_usize(&self);
        let state = CompiledState {
            f: self,

            shapeless,
            id,
            
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<&'a Array, Array, ()> for F
where
    F: FnMut(&Array) -> Array + 'static,
{
    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<&'a Array, Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)(&args[0]);
            vec![result]
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,
            shapeless,
            id,
            
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<&'a Array, Array, Exception> for F
where
    F: FnMut(&Array) -> Result<Array, Exception> + 'static,
{
    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<&'a Array, Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(&args[0])?;
            Ok(vec![result])
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,

            shapeless,
            id,
            
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<(&'a Array, &'a Array), Array, ()> for F
where
    F: FnMut((&Array, &Array)) -> Array + 'static,
{
    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<(&'a Array, &'a Array), Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)((&args[0], &args[1]));
            vec![result]
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,

            shapeless,
            id,
            
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<(&'a Array, &'a Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + 'static,
{
    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<(&'a Array, &'a Array), Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)((&args[0], &args[1]))?;
            Ok(vec![result])
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,

            shapeless,
            id,
            
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<(&'a Array, &'a Array, &'a Array), Array, ()> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Array + 'static,
{
    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<(&'a Array, &'a Array, &'a Array), Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)((&args[0], &args[1], &args[2]));
            vec![result]
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,

            shapeless,
            id,
            
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<(&'a Array, &'a Array, &'a Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + 'static,
{
    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<(&'a Array, &'a Array, &'a Array), Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)((&args[0], &args[1], &args[2]))?;
            Ok(vec![result])
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,

            shapeless,
            id,
            
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

pub trait CallMut<A, O, E> {
    fn call_mut(&mut self, args: A) -> Result<O, Exception>;
}

impl<'a, F, G> CallMut<&'a [Array], Vec<Array>, ()> for Compiled<F, G>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.call_mut(args)
    }
}

impl<'a, F, G> CallMut<&'a [Array], Vec<Array>, Exception> for Compiled<F, G>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.call_mut_fallible(args)
    }
}

impl<'a, F, G> CallMut<&'a Array, Array, ()> for Compiled<F, G>
where
    F: FnMut(&Array) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: &Array) -> Result<Array, Exception> {
        // Is there any way to avoid this shallow clone?
        let args = &[args.clone()];
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<&'a Array, Array, Exception> for Compiled<F, G>
where
    F: FnMut(&Array) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: &Array) -> Result<Array, Exception> {
        // Is there any way to avoid this shallow clone?
        let args = &[args.clone()];
        let result = self.state.call_mut_fallible(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array), Array, ()> for Compiled<F, G>
where
    F: FnMut((&Array, &Array)) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array)) -> Result<Array, Exception> {
        // Is there any way to avoid this shallow clone?
        let args = &[args.0.clone(), args.1.clone()];
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array), Array, Exception> for Compiled<F, G>
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array)) -> Result<Array, Exception> {
        // Is there any way to avoid this shallow clone?
        let args = &[args.0.clone(), args.1.clone()];
        let result = self.state.call_mut_fallible(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array, &'a Array), Array, ()> for Compiled<F, G>
where
    F: FnMut((&Array, &Array, &Array)) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array, &Array)) -> Result<Array, Exception> {
        // Is there any way to avoid this shallow clone?
        let args = &[args.0.clone(), args.1.clone(), args.2.clone()];
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array, &'a Array), Array, Exception>
    for Compiled<F, G>
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array, &Array)) -> Result<Array, Exception> {
        // Is there any way to avoid this shallow clone?
        let args = &[args.0.clone(), args.1.clone(), args.2.clone()];
        let result = self.state.call_mut_fallible(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

/// Returns a compiled function that produces the same output as `f`.
///
/// Please refer to the [swift binding
/// documentation](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/compilation)
/// for more information.
pub fn compile<F, A, O, E>(
    f: F,
    shapeless: impl Into<Option<bool>>,
) -> impl FnMut(A) -> Result<O, Exception>
where
    F: Compile<A, O, E> + 'static,
{
    let shapeless = shapeless.into().unwrap_or(false);
    let mut compiled = f.compile(shapeless);
    move |args| compiled.call_mut(args)
}

#[cfg(test)]
mod tests {
    use core::panic;

    use crate::{
        array,
        error::Exception,
        ops::{multiply, ones},
        Array,
    };

    use super::compile;

    fn example_fn_0(x: f32) -> f32 {
        x + 1.0
    }

    fn example_fn_3(x: f32) -> f32 {
        x + 1.0
    }

    #[test]
    fn test_type_id_to_usize() {
        // We would like to check that different functions that share the same signature can produce
        // different ids

        let example_fn_1 = |x: f32| x + 1.0;
        let example_fn_2 = |x: f32| x + 1.0;

        let mut ids = Vec::new();

        ids.push(super::type_id_to_usize(&example_fn_0));

        let id1 = super::type_id_to_usize(&example_fn_1);
        if ids.contains(&id1) {
            panic!("id1 already exists");
        }
        ids.push(id1);

        let id2 = super::type_id_to_usize(&example_fn_2);
        if ids.contains(&id2) {
            panic!("id2 already exists");
        }
        ids.push(id2);

        let id3 = super::type_id_to_usize(&example_fn_3);
        if ids.contains(&id3) {
            panic!("id3 already exists");
        }
        ids.push(id3);

        assert_eq!(ids.len(), 4);
    }

    #[test]
    fn test_compile() {
        // This unit test is modified from the mlx-swift codebase

        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] * &inputs[1]] };

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();
        let args = [i1, i2];

        // evaluate directly
        let r1 = f(&args).drain(0..1).next().unwrap();

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_error() {
        let f = |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            multiply(&inputs[0], &inputs[1]).map(|x| vec![x])
        };

        // Success case
        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();
        let args = [i1, i2];

        // evaluate directly
        let r1 = f(&args).unwrap().drain(0..1).next().unwrap();

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert_eq!(&r1, &r3);

        // Error case
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let args = [a, b];

        // The cache is keyed by function pointer and argument shapes
        let c = array!([4.0, 5.0, 6.0]);
        let d = array!([7.0, 8.0]);
        let another_args = [c, d];

        // evaluate directly
        let result = f(&args);
        assert!(result.is_err());

        // evaluate compiled
        let mut compiled = compile(f, None);
        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&another_args);
        assert!(result.is_err());
    }

    #[test]
    fn test_compile_with_one_arg() {
        let f = |x: &Array| x * x;

        let i = ones::<f32>(&[20, 20]).unwrap();

        // evaluate directly
        let r1 = f(&i);

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled(&i).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&i).unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_two_args() {
        let f = |(x, y): (&Array, &Array)| x * y;

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();

        // evaluate directly
        let r1 = f((&i1, &i2));

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled((&i1, &i2)).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled((&i1, &i2)).unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_three_args() {
        let f = |(x, y, z): (&Array, &Array, &Array)| x * y * z;

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();
        let i3 = ones::<f32>(&[20, 20]).unwrap();

        // evaluate directly
        let r1 = f((&i1, &i2, &i3));

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled((&i1, &i2, &i3)).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled((&i1, &i2, &i3)).unwrap();
        assert_eq!(&r1, &r3);
    }
}
