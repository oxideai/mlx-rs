extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

mod module_parameters;

/// Derive the `ModuleParameters` trait for a struct. Mark a field with
/// `#[param]` attribute to include it in the parameters. The field type must
/// implement the `mlx_rs::module::Parameter` trait.
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
///     nested: Inner,
///
///     #[param]
///     vec_nested: Vec<Inner>,
///
///     #[param]
///     trait_object: Box<dyn Module>,
///
///     #[param]
///     trait_object_vec: Vec<Box<dyn Module>>,
/// }
///
/// #[derive(ModuleParameters)]
/// struct Inner {
///     #[param]
///     a: Param<Array>,
/// }
/// ```
#[proc_macro_derive(ModuleParameters, attributes(module, param))]
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
