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

// impl<'m, 'a, F, M> Compile<(&'m mut M, &'a [Array]), Vec<Array>, ()> for F
// where
//     F: FnMut((&'m mut M, &'a [Array])) -> Vec<Array> + 'static,
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

// impl<'m, 'a, F, G, M> CallMut<(&'m mut M, &'a [Array]), Vec<Array>, ()> for Compiled<F, G>
// where
//     F: FnMut((&'m mut M, &'a [Array])) -> Vec<Array>,
//     G: FnMut((&'m mut M, &'a [Array])) -> Vec<Array>,
//     M: ModuleParameters,
// {
//     fn call_mut(&mut self, args: (&mut M, &[Array])) -> Result<Vec<Array>, Exception> {
//         todo!()
//     }
// }
