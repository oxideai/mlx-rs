use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::{error::Exception, transforms::compile::{type_id_to_usize, CompiledState}, utils::Updatable, Array};

use super::{update_by_replace_with_ref_to_new_array, Closure, Compiled, Guarded, VectorArray};

pub trait CompileWithState<U, A, O, E> {
    fn compile(self, shapeless: bool) -> impl CallMutWithState<U, A, O, E>;
}

pub trait CallMutWithState<U, A, O, E> {
    fn call_mut(&mut self, module: &mut U, args: A) -> Result<O, Exception>;
}

impl<'a, F, U> CompileWithState<U, &'a [Array], Vec<Array>, ()> for F
where   
    F: FnMut(&mut U, &[Array]) -> Vec<Array> + 'static,
    U: Updatable,
{
    fn compile(self, shapeless: bool) -> impl CallMutWithState<U, &'a [Array], Vec<Array>, ()> {
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

impl<U, F, G> CallMutWithState<U, &[Array], Vec<Array>, ()> for Compiled<F, G> 
where 
    F: FnMut(&mut U, &[Array]) -> Vec<Array>,
    G: FnMut(&mut U, &[Array]) -> Vec<Array>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.call_mut_with_state(state, args)
            .or_else(|_e| {
                // Somehow the mlx_closure_apply may fail on the first call for 
                // certain types of state with the error message:
                // "unordered_map::at: key not found", so we just try again.
                //
                // One type that is known to cause this is a tuple of 
                // `Module` and `Optimizer` eg. `(<Module>, <Optimizer>)`
                self.state.call_mut_with_state(state, args)
            })
    }
}


impl<F> CompiledState<F> {
    fn call_mut_with_state<U>(&mut self, state: &mut U, args: &[Array]) -> Result<Vec<Array>, Exception> 
    where 
        F: FnMut(&mut U, &[Array]) -> Vec<Array>,
        U: Updatable,
    {
        let args_len = args.len();

        let state = Rc::new(RefCell::new(state));

        let f = &mut self.f;

        let state_clone = Rc::clone(&state);
        let inner = move |tracers: &[Array]| -> Vec<Array> {
            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracer_args = &tracers[..args_len];

            // save a snapshot of the inner state
            let saved_state_inputs = state_clone.borrow().updatable_states()
                .iter()
                .map(|array| (*array).clone())
                .collect::<Vec<Array>>();

            // replace the inner state with the tracers
            for (s, tracer) in state_clone.borrow_mut().updatable_states_mut().into_iter().zip(tracers.iter().skip(args_len)) {
                update_by_replace_with_ref_to_new_array(s, tracer);
            }

            // call the function with the tracer arguments and the state holding tracers
            let mut result = (f)(*state_clone.borrow_mut(), tracer_args);

            // recapture the state as it may have changed
            let mut state_output_tracers = state_clone.borrow().updatable_states()
                .iter()
                .map(|array| (*array).clone())
                .collect::<Vec<Array>>();

            // put the original values back in the state
            for (s, saved) in state_clone.borrow_mut().updatable_states_mut().into_iter().zip(saved_state_inputs) {
                update_by_replace_with_ref_to_new_array(s, &saved);
            }

            // return the result of the function and the state
            result.append(&mut state_output_tracers);

            result
        };

        let inner_closure = Closure::new(inner);

        call_mut_with_module_inner(
            inner_closure,
            self.id,
            self.shapeless,
            state,
            args,
        )
    }
}


#[inline]
fn call_mut_with_module_inner<U>(
    inner_closure: Closure,
    fun_id: usize,
    shapeless: bool,
    state: Rc<RefCell<&mut U>>,
    args: &[Array],
) -> crate::error::Result<Vec<Array>> 
where 
    U: Updatable
{
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

    let (state_params_len, inner_inputs_vector) = {
        let borrow = state.borrow();
        let state_params = borrow.updatable_states();
        let state_params_len = state_params.len();
        let inner_inputs_vector = VectorArray::try_from_iter(args.iter().chain(state_params.into_iter()))?;
        (state_params_len, inner_inputs_vector)
    };

    // will compile the function (if needed) and evaluate the
    // compiled graph
    let result_vector = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
    })?;
    let result_plus_state_output: Vec<Array> = result_vector.try_into_values()?;

    // push the stateOutput into the state
    let result_plus_state_output_len = result_plus_state_output.len();
    let suffix_len = result_plus_state_output_len - state_params_len;
    for (s, new_values) in state
        .borrow_mut()
        .updatable_states_mut()
        .into_iter()
        .zip(result_plus_state_output[suffix_len..].iter())
    {
        update_by_replace_with_ref_to_new_array(s, new_values);
    }

    let result_len = result_plus_state_output.len()
        - state_params_len;
    Ok(result_plus_state_output
        .into_iter()
        .take(result_len)
        .collect())
}

pub fn compile_with_state<F, U, A, O, E>(
    f: F,
    shapeless: impl Into<Option<bool>>,
) -> impl FnMut(&mut U, A) -> Result<O, Exception> 
where 
    F: CompileWithState<U, A, O, E> + 'static,
    U: Updatable,
{
    let shapeless = shapeless.into().unwrap_or(false);
    let mut compiled = f.compile(shapeless);
    move |module, args| compiled.call_mut(module, args)
}