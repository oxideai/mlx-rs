//! Compilation of functions.

use std::{
    cell::RefCell,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    marker::PhantomData,
    rc::Rc,
};

use mlx_sys::{
    mlx_closure_apply, mlx_detail_compile, mlx_detail_compile_clear_cache,
    mlx_detail_compile_erase, mlx_disable_compile, mlx_enable_compile,
};

use crate::{
    error::Exception,
    utils::{guard::Guarded, Closure, VectorArray},
    Array,
};

use super::get_and_clear_closure_error;

/// Globally enable the compilation of functions.
///
/// Default is enabled.
pub fn enable_compile() {
    unsafe {
        mlx_enable_compile();
    }
}

/// Globally disable the compilation of functions.
///
/// Default is enabled.
pub fn disable_compile() {
    unsafe {
        mlx_disable_compile();
    }
}

/// A trait for functions that can be compiled.
pub trait Compile<'a, Args, Output, Err>: Sized {
    /// Compile the function.
    fn compile(
        self,
        inputs: Option<&'a mut [Array]>,
        outputs: Option<&'a mut [Array]>,
        shapeless: bool,
    ) -> impl CallMut<'a, Args, Output, Err>;
}

impl<'a, F> Compile<'a, &'a [Array], Vec<Array>, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    fn compile(
        self,
        inputs: Option<&'a mut [Array]>,
        outputs: Option<&'a mut [Array]>,
        shapeless: bool,
    ) -> impl CallMut<'a, &'a [Array], Vec<Array>, ()> {
        let id = type_id_to_usize(&self);
        let state = CompiledState {
            f: self,
            inputs,
            outputs,
            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<'a, &'a [Array], Vec<Array>, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    fn compile(
        self,
        inputs: Option<&'a mut [Array]>,
        outputs: Option<&'a mut [Array]>,
        shapeless: bool,
    ) -> impl CallMut<'a, &'a [Array], Vec<Array>, Exception> {
        let id = type_id_to_usize(&self);
        let state = CompiledState {
            f: self,
            inputs,
            outputs,
            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<'a, &'a Array, Array, ()> for F
where
    F: FnMut(&Array) -> Array + 'static,
{
    fn compile(
        mut self,
        inputs: Option<&'a mut [Array]>,
        outputs: Option<&'a mut [Array]>,
        shapeless: bool,
    ) -> impl CallMut<'a, &'a Array, Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)(&args[0]);
            vec![result]
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,
            inputs,
            outputs,
            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<'a, &'a Array, Array, Exception> for F
where
    F: FnMut(&Array) -> Result<Array, Exception> + 'static,
{
    fn compile(
        mut self,
        inputs: Option<&'a mut [Array]>,
        outputs: Option<&'a mut [Array]>,
        shapeless: bool,
    ) -> impl CallMut<'a, &'a Array, Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(&args[0])?;
            Ok(vec![result])
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,
            inputs,
            outputs,
            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<'a, (&'a Array, &'a Array), Array, ()> for F
where
    F: FnMut((&Array, &Array)) -> Array + 'static,
{
    fn compile(
        mut self,
        inputs: Option<&'a mut [Array]>,
        outputs: Option<&'a mut [Array]>,
        shapeless: bool,
    ) -> impl CallMut<'a, (&'a Array, &'a Array), Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)((&args[0], &args[1]));
            vec![result]
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,
            inputs,
            outputs,
            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<'a, (&'a Array, &'a Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + 'static,
{
    fn compile(
        mut self,
        inputs: Option<&'a mut [Array]>,
        outputs: Option<&'a mut [Array]>,
        shapeless: bool,
    ) -> impl CallMut<'a, (&'a Array, &'a Array), Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)((&args[0], &args[1]))?;
            Ok(vec![result])
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,
            inputs,
            outputs,
            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<'a, (&'a Array, &'a Array, &'a Array), Array, ()> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Array + 'static,
{
    fn compile(
        mut self,
        inputs: Option<&'a mut [Array]>,
        outputs: Option<&'a mut [Array]>,
        shapeless: bool,
    ) -> impl CallMut<'a, (&'a Array, &'a Array, &'a Array), Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)((&args[0], &args[1], &args[2]));
            vec![result]
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,
            inputs,
            outputs,
            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<'a, F> Compile<'a, (&'a Array, &'a Array, &'a Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + 'static,
{
    fn compile(
        mut self,
        inputs: Option<&'a mut [Array]>,
        outputs: Option<&'a mut [Array]>,
        shapeless: bool,
    ) -> impl CallMut<'a, (&'a Array, &'a Array, &'a Array), Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)((&args[0], &args[1], &args[2]))?;
            Ok(vec![result])
        };
        let id = type_id_to_usize(&f);
        let state = CompiledState {
            f,
            inputs,
            outputs,
            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

/// A trait for a compiled function that can be called.
pub trait CallMut<'a, Args, Output, Err> {
    /// Call the compiled function.
    fn call_mut(&mut self, args: Args) -> Result<Output, Exception>;
}

/// A compiled function that can be called.
#[derive(Debug)]
pub struct Compiled<'a, F, G> {
    f_marker: std::marker::PhantomData<F>,
    state: CompiledState<'a, G>,
}

impl<'a, F, G> CallMut<'a, &'a [Array], Vec<Array>, ()> for Compiled<'a, F, G>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.call_mut(args)
    }
}

impl<'a, F, G> CallMut<'a, &'a [Array], Vec<Array>, Exception> for Compiled<'a, F, G>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.call_mut_fallible(args)
    }
}

impl<'a, F, G> CallMut<'a, &'a Array, Array, ()> for Compiled<'a, F, G>
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

impl<'a, F, G> CallMut<'a, &'a Array, Array, Exception> for Compiled<'a, F, G>
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

impl<'a, F, G> CallMut<'a, (&'a Array, &'a Array), Array, ()> for Compiled<'a, F, G>
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

impl<'a, F, G> CallMut<'a, (&'a Array, &'a Array), Array, Exception> for Compiled<'a, F, G>
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

impl<'a, F, G> CallMut<'a, (&'a Array, &'a Array, &'a Array), Array, ()> for Compiled<'a, F, G>
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

impl<'a, F, G> CallMut<'a, (&'a Array, &'a Array, &'a Array), Array, Exception>
    for Compiled<'a, F, G>
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

#[derive(Debug)]
struct CompiledState<'a, F>
where
    F: 'a,
{
    f: F,
    inputs: Option<&'a mut [Array]>,
    outputs: Option<&'a mut [Array]>,
    shapeless: bool,
    id: usize,
}

#[inline]
fn call_mut_inner(
    inner_closure: Closure,
    fun_id: usize,
    shapeless: bool,
    state_inputs: Rc<RefCell<&mut Option<&mut [Array]>>>,
    state_outputs: Rc<RefCell<&mut Option<&mut [Array]>>>,
    args: &[Array],
) -> crate::error::Result<Vec<Array>> {
    // note: this will use the cached compile (via the id)
    // but will be able to re-evaluate with fresh state if needed
    let compiled = Closure::try_from_op(|res| unsafe {
        let constants = &[];
        mlx_detail_compile(
            res,
            inner_closure.as_ptr(),
            fun_id,
            shapeless,
            constants.as_ptr(),
            0,
        )
    })?;

    let inner_inputs_vector = match state_inputs.borrow().as_ref() {
        Some(s) => VectorArray::try_from_iter(args.iter().chain(s.iter()))?,
        None => VectorArray::try_from_iter(args.iter())?,
    };

    // will compile the function (if needed) and evaluate the
    // compiled graph
    let result_vector = VectorArray::try_from_op(|res| unsafe {
        mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
    })
    .map_err(|e| match get_and_clear_closure_error() {
        Some(err) => err,
        None => e,
    })?;
    let result_plus_state_output: Vec<Array> = result_vector.try_into_values()?;

    // push the stateOutput into the state
    if let Some(outputs) = state_outputs.borrow_mut().as_mut() {
        let result_plus_state_output_len = result_plus_state_output.len();
        let state_output_len = outputs.len();
        let suffix_len = result_plus_state_output_len - state_output_len;
        for (s, new_values) in outputs
            .iter_mut()
            .zip(result_plus_state_output[suffix_len..].iter())
        {
            update_by_replace_with_ref_to_new_array(s, new_values);
        }
    }

    let result_len = result_plus_state_output.len()
        - state_outputs
            .borrow()
            .as_ref()
            .map(|x| x.len())
            .unwrap_or(0);
    Ok(result_plus_state_output
        .into_iter()
        .take(result_len)
        .collect())
}

impl<'a, F> CompiledState<'a, F> {
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Vec<Array> + 'a,
    {
        let args_len = args.len();
        let state_inputs = Rc::new(RefCell::new(&mut self.inputs));
        let state_outputs = Rc::new(RefCell::new(&mut self.outputs));
        let f = &mut self.f;

        let state_inputs_clone = Rc::clone(&state_inputs);
        let state_outputs_clone = Rc::clone(&state_outputs);
        let inner = move |tracers: &[Array]| -> Vec<Array> {
            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracer_args = &tracers[..args_len];

            // save a snapshot of the inner state
            let saved_state_inputs: Option<Vec<Array>> = state_inputs_clone
                .borrow()
                .as_ref()
                .map(|inputs| inputs.iter().map(Clone::clone).collect());

            // replace the inner state with the tracers
            if let Some(inputs) = state_inputs_clone.borrow_mut().as_mut() {
                for (s, tracer) in inputs.iter_mut().zip(tracers.iter().skip(args_len)) {
                    update_by_replace_with_ref_to_new_array(s, tracer);
                }
            }

            // call the function with the tracer arguments and the state holding tracers
            let mut result = (f)(tracer_args);

            // recapture the state as it may have changed
            let state_output_tracers: Option<Vec<Array>> = state_outputs_clone
                .borrow()
                .as_ref()
                .map(|outputs| outputs.iter().map(Clone::clone).collect());

            // put the original values back in the state
            if let Some(inputs) = state_inputs_clone.borrow_mut().as_mut() {
                for (s, saved) in inputs.iter_mut().zip(saved_state_inputs.unwrap()) {
                    update_by_replace_with_ref_to_new_array(s, &saved);
                }
            }

            // return the result of the function and the state
            if let Some(mut state_output_tracers) = state_output_tracers {
                result.append(&mut state_output_tracers);
            }

            result
        };

        let inner_closure = Closure::new(inner);

        call_mut_inner(
            inner_closure,
            self.id,
            self.shapeless,
            state_inputs,
            state_outputs,
            args,
        )
    }

    fn call_mut_fallible(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
    {
        let args_len = args.len();
        let state_inputs = Rc::new(RefCell::new(&mut self.inputs));
        let state_outputs = Rc::new(RefCell::new(&mut self.outputs));
        let f = &mut self.f;

        let state_inputs_clone = Rc::clone(&state_inputs);
        let state_outputs_clone = Rc::clone(&state_outputs);
        let inner = move |tracers: &[Array]| -> Result<Vec<Array>, Exception> {
            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracer_args = &tracers[..args_len];

            // save a snapshot of the inner state
            let saved_state_inputs: Option<Vec<Array>> = state_inputs_clone
                .borrow()
                .as_ref()
                .map(|inputs| inputs.iter().map(Clone::clone).collect());

            // replace the inner state with the tracers
            if let Some(inputs) = state_inputs_clone.borrow_mut().as_mut() {
                for (s, tracer) in inputs.iter_mut().zip(tracers.iter().skip(args_len)) {
                    update_by_replace_with_ref_to_new_array(s, tracer);
                }
            }

            // call the function with the tracer arguments and the state holding tracers
            let mut result = (f)(tracer_args);

            // recapture the state as it may have changed
            let state_output_tracers: Option<Vec<Array>> = state_outputs_clone
                .borrow()
                .as_ref()
                .map(|outputs| outputs.iter().map(Clone::clone).collect());

            // put the original values back in the state
            if let Some(inputs) = state_inputs_clone.borrow_mut().as_mut() {
                for (s, saved) in inputs.iter_mut().zip(saved_state_inputs.unwrap()) {
                    update_by_replace_with_ref_to_new_array(s, &saved);
                }
            }

            // return the result of the function and the state
            if let Some(mut state_output_tracers) = state_output_tracers {
                result = result.map(|mut r| {
                    r.append(&mut state_output_tracers);
                    r
                });
            }

            result
        };

        let inner_closure = Closure::new_fallible(inner);

        call_mut_inner(
            inner_closure,
            self.id,
            self.shapeless,
            state_inputs,
            state_outputs,
            args,
        )
    }
}

impl<F> Drop for CompiledState<'_, F> {
    fn drop(&mut self) {
        unsafe {
            // remove the compiled structure from the back end
            mlx_detail_compile_erase(self.id);
        }
    }
}

fn type_id_to_usize<T>(_val: &T) -> usize
where
    T: 'static,
{
    // hash type id to usize
    let type_id = std::any::TypeId::of::<T>();
    let mut hasher = DefaultHasher::new();
    type_id.hash(&mut hasher);
    hasher.finish() as usize
}

fn update_by_replace_with_ref_to_new_array(src: &mut Array, new_array: &Array) {
    unsafe {
        mlx_sys::mlx_array_set(&mut src.c_array as *mut _, new_array.c_array);
    }
}

/// Returns a compiled function that produces the same output as `f`.
///
/// Please refer to the [swift binding
/// documentation](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/compilation)
/// for more information.
pub fn compile<'a, F, Args, Output, Err>(
    f: F,
    shapeless: Option<bool>,
    inputs: Option<&'a mut [Array]>,
    outputs: Option<&'a mut [Array]>,
) -> impl FnMut(Args) -> Result<Output, Exception> + 'a
where
    F: Compile<'a, Args, Output, Err> + 'static,
    Args: 'a,
    Output: 'a,
    Err: 'a,
{
    let shapeless = shapeless.unwrap_or(false);
    let mut compiled = f.compile(inputs, outputs, shapeless);
    move |args| compiled.call_mut(args)
}

/// Clear the memory cache.
pub fn clear_cache() {
    unsafe {
        mlx_detail_compile_clear_cache();
    }
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
        let mut compiled = compile(f, None, None, None);
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
        let mut compiled = compile(f, None, None, None);
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
        let mut compiled = compile(f, None, None, None);
        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&another_args);
        assert!(result.is_err());

        // Check that the error message is not just "mlx_closure returned a non-zero value"
        let error = result.unwrap_err();
        assert!(!error.what().contains("non-zero value"));
    }

    #[test]
    fn test_compile_with_one_arg() {
        let f = |x: &Array| x * x;

        let i = ones::<f32>(&[20, 20]).unwrap();

        // evaluate directly
        let r1 = f(&i);

        // evaluate compiled
        let mut compiled = compile(f, None, None, None);
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
        let mut compiled = compile(f, None, None, None);
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
        let mut compiled = compile(f, None, None, None);
        let r2 = compiled((&i1, &i2, &i3)).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled((&i1, &i2, &i3)).unwrap();
        assert_eq!(&r1, &r3);
    }
}
