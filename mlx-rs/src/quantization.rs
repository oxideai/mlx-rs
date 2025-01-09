//! Traits for quantization

// use std::marker::PhantomData;

// use crate::{module::Module, Array};

// /// Marker trait for a quantizable module.z
// pub trait QuantizableModule<Args> {
//     /// The quantized version of the module.
//     type Quantized: Module<Args>;

//     /// Convert the module into a quantized version.
//     fn into_quantized(self) -> Self::Quantized;
// }

// /// A wrapper for a quantizable module.
// #[derive(Debug, Clone)]
// pub enum Quantizable<M, Q> 
// where 
//     for<'a> M: QuantizableModule<Quantized = Q>,
// {
//     /// The original module.
//     Original(M),

//     /// The quantized version of the module.
//     Quantized(Q),
// }