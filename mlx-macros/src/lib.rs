extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

mod module_parameters;

/// Derive the `ModuleParameters` trait for a struct. Mark a field with `#[param]` attribute to
/// include it in the parameters. The field type must implement the `Parameter` trait defined in
/// `mlx-nn-module` crate.
///
/// Make sure to include `mlx-nn-module` as a dependency in your `Cargo.toml`.
///
/// # Example
///
/// ```rust, ignore
/// use mlx_macros::ModuleParameters;
/// use mlx_rs::module::{ModuleParameters, Param};
///
/// #[derive(ModuleParameters)]
/// struct Example {
///     #[param]
///     regular: Param<Array>,
///
///     #[param]
///     optional: Param<Option<Array>>,
///
///     #[param]
///     nested: Param<Inner>,
///
///     #[param]
///     vec_nested: Param<Vec<Inner>>,
///
///     #[param]
///     trait_object: Param<Box<dyn Module>>,
///
///     #[param]
///     trait_object_vec: Param<Vec<Box<dyn Module>>>,
/// }
///
/// #[derive(ModuleParameters)]
/// struct Inner {
///     #[param]
///     a: Param<Array>,
/// }
/// ```
#[proc_macro_derive(ModuleParameters, attributes(param))]
pub fn derive_module_parameters(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let module_param_impl = module_parameters::expand_module_parameters(&input).unwrap();

    let output = quote! {
        const _: () = {
            extern crate mlx_rs as _mlx_rs;
            #module_param_impl
        };
    };
    TokenStream::from(output)
}

/// Derive the `MaybeQuantizable` trait for a struct.
/// 
/// TODO: Two approaches:
/// 
/// 1. Trait object based. Using traits to eventually return something that `impl Module`.
/// 2. Macro based. Using a macro to generate a new struct if any field is quantizable.
/// 
/// Which one is better?
#[proc_macro_derive(MaybeQuantizable)]
pub fn derive_maybe_quantizable(input: TokenStream) -> TokenStream {

    todo!()
}