use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use libc::stat;

use crate::{error::Exception, module::ModuleParameters, transforms::compile::{type_id_to_usize, update_by_replace_with_ref_to_new_array, CompiledState}, utils::{Closure, Updatable}, Array};

use super::{CallMut, Compile, Compiled, Guarded, VectorArray};

// /// # Generics:
// /// 
// /// - `U`: The type of the updatable
// /// - `A`: The type of the array arguments
// /// - `O`: The type of the output
// /// - `E`: The type of the error
// pub trait CompileModule<M, A, O, E> {
//     fn compile_updatable(self, module: &mut M, shapeless: bool) -> impl CallMutUpdatable<M, A, O, E>;
// }

// impl<'a, F, M> CompileModule<M, &'a [Array], Vec<Array>, ()> for F
// where 
//     F: FnMut(&'a mut M, &'a [Array]) -> Vec<Array> + 'static,
//     M: ModuleParameters + 'a,
// {
//     fn compile_updatable(self, module: &mut M, shapeless: bool) -> impl CallMutUpdatable<M, &'a [Array], Vec<Array>, ()> {
//         let id = type_id_to_usize(&self);
//         let updatable_state = module.updatable_parameters_mut();
//         let state = CompiledState {
//             f: self,
//             state: Some(updatable_state),
//             shapeless,
//             id,
//         };
//         Compiled {
//             f_marker: PhantomData::<F>,
//             state,
//         }
//     }
// }

// pub trait CallMutUpdatable<M, A, O, E> {
//     fn call_mut_updatable(&mut self, module: &mut M, args: A) -> Result<O, Exception>;
// }

// impl<'a, F, G, M> CallMutUpdatable<M, &'a [Array], Vec<Array>, ()> for Compiled<'a, F, G>
// where 
//     F: FnMut(&'a mut M, &'a [Array]) -> Vec<Array>,
//     G: FnMut(&'a mut M, &'a [Array]) -> Vec<Array>,
//     M: ModuleParameters + 'a,
// {
//     fn call_mut_updatable(&mut self, module: &mut M, args: &'a [Array]) -> Result<Vec<Array>, Exception> {
//         todo!()
//     }
// }

// impl<'m, 'a, F, M> Compile<(&'m mut M, &'a [Array]), Vec<Array>, ()> for F
// where
//     for<'m_, 'a_> F: FnMut((&'m_ mut M, &'a_ [Array])) -> Vec<Array> + 'static,
//     M: ModuleParameters,
// {
//     fn compile(self, shapeless: bool) -> impl CallMut<(&'m mut M, &'a [Array]), Vec<Array>, ()> {
//         let id = type_id_to_usize(&self);
//         let state = CompiledState {
//             f: self,
//             shapeless,
//             id,
//         };
//         Compiled {
//             f_marker: PhantomData::<F>,
//             state,
//         }
//     }
// }

// impl<F> CompiledState<F> {
//     fn call_mut_with_module<'m, 'a, M>(&mut self, module: &'m mut M, args: &'a [Array]) -> Result<Vec<Array>, Exception> 
//     where 
//         for<'m_, 'a_> F: FnMut((&'m_ mut M, &'a_ [Array])) -> Vec<Array> + 'static,
//         M: ModuleParameters + 'm,
//     {
//         let args_len = args.len();

//         // TODO: do we need both inputs and outputs? They are the same in this case
//         let mut inputs: Vec<Array> = module.updatable_parameters()
//             .iter()
//             .map(|array| (*array).clone())
//             .collect();
//         let mut outputs: Vec<Array> = module.updatable_parameters()
//             .iter()
//             .map(|array| (*array).clone())
//             .collect();

//         let state_inputs = Rc::new(RefCell::new(&mut inputs));
//         let state_outputs = Rc::new(RefCell::new(&mut outputs));
//         let f = &mut self.f;

//         let state_inputs_clone = Rc::clone(&state_inputs);
//         let state_outputs_clone = Rc::clone(&state_outputs);
//         let inner = move |tracers: &[Array]| -> Vec<Array> {
//             // put the tracers in their appropriate places:
//             // - arguments to the function
//             // - inner state

//             let tracer_args = &tracers[..args_len];

//             // save a snapshot of the inner state
//             let saved_state_inputs: Vec<Array> = state_inputs_clone
//                 .borrow()
//                 .iter().map(|a| (*a).clone()).collect();

//             // replace the inner state with the tracers
//             for (s, tracer) in state_inputs_clone.borrow_mut().iter_mut().zip(tracers.iter().skip(args_len)) {
//                 update_by_replace_with_ref_to_new_array(s, tracer);
//             }

//             // call the function with the tracer arguments and the state holding tracers
//             let mut result = (f)((module, tracer_args));

//             // recapture the state as it may have changed
//             let mut state_output_tracers: Vec<Array> = state_outputs_clone
//                 .borrow()
//                 .iter().map(|a| (*a).clone()).collect();

//             // put the original values back in the state
//             for (s, saved) in state_inputs_clone.borrow_mut().iter_mut().zip(saved_state_inputs) {
//                 update_by_replace_with_ref_to_new_array(s, &saved);
//             }

//             // return the result of the function and the state
//             result.append(&mut state_output_tracers);

//             result
//         };

//         let inner_closure = Closure::new(inner);

//         call_mut_with_module_inner(
//             inner_closure,
//             self.id,
//             self.shapeless,
//             state_inputs,
//             state_outputs,
//             args,
//         )
//     }
// }


// #[inline]
// fn call_mut_with_module_inner(
//     inner_closure: Closure,
//     fun_id: usize,
//     shapeless: bool,
//     state_inputs: Rc<RefCell<&mut Vec<Array>>>,
//     state_outputs: Rc<RefCell<&mut Vec<Array>>>,
//     args: &[Array],
// ) -> crate::error::Result<Vec<Array>> {
//     // note: this will use the cached compile (via the id)
//     // but will be able to re-evaluate with fresh state if needed
//     let compiled = Closure::try_from_op(|res| unsafe {
//         let constants = &[];
//         mlx_sys::mlx_detail_compile(
//             res,
//             inner_closure.as_ptr(),
//             fun_id,
//             shapeless,
//             constants.as_ptr(),
//             0,
//         )
//     })?;

//     let inner_inputs_vector = VectorArray::try_from_iter(args.iter().chain(state_inputs.borrow().iter()))?;

//     // will compile the function (if needed) and evaluate the
//     // compiled graph
//     let result_vector = VectorArray::try_from_op(|res| unsafe {
//         mlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
//     })?;
//     let result_plus_state_output: Vec<Array> = result_vector.try_into_values()?;

//     // push the stateOutput into the state
//     let result_plus_state_output_len = result_plus_state_output.len();
//     let state_output_len = state_outputs.borrow().len();
//     let suffix_len = result_plus_state_output_len - state_output_len;
//     for (s, new_values) in state_outputs
//         .borrow_mut()
//         .iter_mut()
//         .zip(result_plus_state_output[suffix_len..].iter())
//     {
//         update_by_replace_with_ref_to_new_array(s, new_values);
//     }

//     let result_len = result_plus_state_output.len()
//         - state_outputs
//             .borrow()
//             .len();
//     Ok(result_plus_state_output
//         .into_iter()
//         .take(result_len)
//         .collect())
// }

// impl<'m, 'a, F, G, M> CallMut<(&'m mut M, &'a [Array]), Vec<Array>, ()> for Compiled<F, G>
// where
//     for<'m_, 'a_> F: FnMut((&'m_ mut M, &'a_ [Array])) -> Vec<Array> + 'static,
//     for<'m_, 'a_> G: FnMut((&'m_ mut M, &'a_ [Array])) -> Vec<Array> + 'static,
//     M: ModuleParameters,
// {
//     fn call_mut(&mut self, args: (&mut M, &[Array])) -> Result<Vec<Array>, Exception> {
//         self.state.call_mut_with_module(args.0, args.1)
//     }
// }
