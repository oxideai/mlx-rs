use std::{
    cell::RefCell, hash::{DefaultHasher, Hash, Hasher}, marker::PhantomData, rc::Rc
};

use super::{Closure, Guarded, VectorArray};
use crate::{error::Exception, Array};

#[allow(clippy::module_inception)]
mod compile;
mod compile_updatable;

pub use compile::*;
pub use compile_updatable::*;

#[derive(Debug)]
pub struct Compiled<F, G> {
    f_marker: std::marker::PhantomData<F>,
    state: CompiledState<G>,
}

#[derive(Debug)]
struct CompiledState<F> {
    f: F,
    // inputs: Option<Vec<&'a Array>>,
    // outputs: Option<Vec<&'a Array>>,
    // state: Option<Vec<&'a mut Array>>,
    shapeless: bool,
    id: usize,
    // lt_marker: PhantomData<&'a ()>, // TODO: is this needed?
}

#[inline]
fn call_mut_inner(
    inner_closure: Closure,
    fun_id: usize,
    shapeless: bool,
    // state_inputs: Rc<RefCell<&mut Option<Vec<Array>>>>,
    // state_outputs: Rc<RefCell<&mut Option<Vec<Array>>>>,
    args: &[Array],
) -> crate::error::Result<Vec<Array>> {
    // note: this will use the cached compile (via the id)
    // but will be able to re-evaluate with fresh state if needed
    let compiled = Closure::try_from_op(|res| unsafe {
        let constants = &[];
        mlx_sys::mlx_detail_compile(
            res,
            inner_closure.as_ptr(),
            fun_id,
            shapeless,
            constants.as_ptr(),
            0,
        )
    })?;

    // let inner_inputs_vector = match state_inputs.borrow().as_ref() {
    //     Some(s) => VectorArray::try_from_iter(args.iter().chain(s.iter()))?,
    //     None => VectorArray::try_from_iter(args.iter())?,
    // };
    let inner_inputs_vector = VectorArray::try_from_iter(args.iter())?;

    // will compile the function (if needed) and evaluate the
    // compiled graph
    let result_vector = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
    })?;
    let result_plus_state_output: Vec<Array> = result_vector.try_into_values()?;

    // // push the stateOutput into the state
    // if let Some(outputs) = state_outputs.borrow_mut().as_mut() {
    //     let result_plus_state_output_len = result_plus_state_output.len();
    //     let state_output_len = outputs.len();
    //     let suffix_len = result_plus_state_output_len - state_output_len;
    //     for (s, new_values) in outputs
    //         .iter_mut()
    //         .zip(result_plus_state_output[suffix_len..].iter())
    //     {
    //         update_by_replace_with_ref_to_new_array(s, new_values);
    //     }
    // }

    // let result_len = result_plus_state_output.len()
    //     - state_outputs
    //         .borrow()
    //         .as_ref()
    //         .map(|x| x.len())
    //         .unwrap_or(0);
    let result_len = result_plus_state_output.len();
    Ok(result_plus_state_output
        .into_iter()
        .take(result_len)
        .collect())
}

impl<F> CompiledState<F> {
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Vec<Array>,
    {
        let args_len = args.len();

        // let mut inputs: Option<Vec<Array>> = self
        //     .state
        //     .as_ref()
        //     .map(|s| s.iter().map(|a| (*a).clone()).collect::<Vec<_>>());
        // let mut outputs: Option<Vec<Array>> = self
        //     .state
        //     .as_ref()
        //     .map(|s| s.iter().map(|a| (*a).clone()).collect::<Vec<_>>());

        // let state_inputs = Rc::new(RefCell::new(&mut inputs));
        // let state_outputs = Rc::new(RefCell::new(&mut outputs));
        let f = &mut self.f;

        // let state_inputs_clone = Rc::clone(&state_inputs);
        // let state_outputs_clone = Rc::clone(&state_outputs);
        let inner = move |tracers: &[Array]| -> Vec<Array> {
            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracer_args = &tracers[..args_len];

            // // save a snapshot of the inner state
            // let saved_state_inputs: Option<Vec<Array>> = state_inputs_clone
            //     .borrow()
            //     .as_ref()
            //     .map(|inputs| inputs.iter().map(|a| (*a).clone()).collect());

            // // replace the inner state with the tracers
            // if let Some(inputs) = state_inputs_clone.borrow_mut().as_mut() {
            //     for (s, tracer) in inputs.iter_mut().zip(tracers.iter().skip(args_len)) {
            //         update_by_replace_with_ref_to_new_array(s, tracer);
            //     }
            // }

            // call the function with the tracer arguments and the state holding tracers
            let mut result = (f)(tracer_args);

            // // recapture the state as it may have changed
            // let state_output_tracers: Option<Vec<Array>> = state_outputs_clone
            //     .borrow()
            //     .as_ref()
            //     .map(|outputs| outputs.iter().map(|a| (*a).clone()).collect());

            // // put the original values back in the state
            // if let Some(inputs) = state_inputs_clone.borrow_mut().as_mut() {
            //     for (s, saved) in inputs.iter_mut().zip(saved_state_inputs.unwrap()) {
            //         update_by_replace_with_ref_to_new_array(s, &saved);
            //     }
            // }

            // // return the result of the function and the state
            // if let Some(mut state_output_tracers) = state_output_tracers {
            //     result.append(&mut state_output_tracers);
            // }

            result
        };

        let inner_closure = Closure::new(inner);

        call_mut_inner(
            inner_closure,
            self.id,
            self.shapeless,
            // state_inputs,
            // state_outputs,
            args,
        )
    }

    fn call_mut_fallible(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception>,
    {
        let args_len = args.len();

        // let mut inputs: Option<Vec<Array>> = self
        //     .state
        //     .as_ref()
        //     .map(|s| s.iter().map(|a| (*a).clone()).collect::<Vec<_>>());
        // let mut outputs: Option<Vec<Array>> = self
        //     .state
        //     .as_ref()
        //     .map(|s| s.iter().map(|a| (*a).clone()).collect::<Vec<_>>());

        // let state_inputs = Rc::new(RefCell::new(&mut inputs));
        // let state_outputs = Rc::new(RefCell::new(&mut outputs));

        // let state_inputs = Rc::new(RefCell::new(&mut self.inputs));
        // let state_outputs = Rc::new(RefCell::new(&mut self.outputs));
        let f = &mut self.f;

        // let state_inputs_clone = Rc::clone(&state_inputs);
        // let state_outputs_clone = Rc::clone(&state_outputs);
        let inner = move |tracers: &[Array]| -> Result<Vec<Array>, Exception> {
            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracer_args = &tracers[..args_len];

            // // save a snapshot of the inner state
            // let saved_state_inputs: Option<Vec<Array>> = state_inputs_clone
            //     .borrow()
            //     .as_ref()
            //     .map(|inputs| inputs.iter().map(|a| (*a).clone()).collect());

            // // replace the inner state with the tracers
            // if let Some(inputs) = state_inputs_clone.borrow_mut().as_mut() {
            //     for (s, tracer) in inputs.iter_mut().zip(tracers.iter().skip(args_len)) {
            //         update_by_replace_with_ref_to_new_array(s, tracer);
            //     }
            // }

            // call the function with the tracer arguments and the state holding tracers
            let mut result = (f)(tracer_args);

            // // recapture the state as it may have changed
            // let state_output_tracers: Option<Vec<Array>> = state_outputs_clone
            //     .borrow()
            //     .as_ref()
            //     .map(|outputs| outputs.iter().map(|a| (*a).clone()).collect());

            // // put the original values back in the state
            // if let Some(inputs) = state_inputs_clone.borrow_mut().as_mut() {
            //     for (s, saved) in inputs.iter_mut().zip(saved_state_inputs.unwrap()) {
            //         update_by_replace_with_ref_to_new_array(s, &saved);
            //     }
            // }

            // // return the result of the function and the state
            // if let Some(mut state_output_tracers) = state_output_tracers {
            //     result = result.map(|mut r| {
            //         r.append(&mut state_output_tracers);
            //         r
            //     });
            // }

            result
        };

        let inner_closure = Closure::new_fallible(inner);

        call_mut_inner(
            inner_closure,
            self.id,
            self.shapeless,
            // state_inputs,
            // state_outputs,
            args,
        )
    }
}

impl<F> Drop for CompiledState<F> {
    fn drop(&mut self) {
        unsafe {
            // remove the compiled structure from the back end
            mlx_sys::mlx_detail_compile_erase(self.id);
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
