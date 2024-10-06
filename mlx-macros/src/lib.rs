extern crate proc_macro;
use darling::FromMeta;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, parse_quote, DeriveInput, FnArg, ItemFn, ItemStruct, Pat};

mod module_parameters;
mod option_builder;

#[derive(Debug, FromMeta)]
enum DeviceType {
    Cpu,
    Gpu,
}

#[derive(Debug)]
struct DefaultDeviceInput {
    device: DeviceType,
}

impl FromMeta for DefaultDeviceInput {
    fn from_meta(meta: &syn::Meta) -> darling::Result<Self> {
        let syn::Meta::NameValue(meta_name_value) = meta else {
            return Err(darling::Error::unsupported_format(
                "expected a name-value attribute",
            ));
        };

        let ident = meta_name_value.path.get_ident().unwrap();
        assert_eq!(ident, "device", "expected `device`");

        let device = DeviceType::from_expr(&meta_name_value.value)?;

        Ok(DefaultDeviceInput { device })
    }
}

#[proc_macro_attribute]
pub fn default_device(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = if !attr.is_empty() {
        let meta = syn::parse_macro_input!(attr as syn::Meta);
        Some(DefaultDeviceInput::from_meta(&meta).unwrap())
    } else {
        None
    };

    let mut input_fn = parse_macro_input!(item as ItemFn);
    let original_fn = input_fn.clone();

    // Ensure function name convention
    if !input_fn.sig.ident.to_string().contains("_device") {
        panic!("Function name must end with '_device'");
    }
    let new_fn_name = format_ident!("{}", &input_fn.sig.ident.to_string().replace("_device", ""));
    input_fn.sig.ident = new_fn_name;

    // Filter out the `stream` parameter and reconstruct the Punctuated collection
    let filtered_inputs = input_fn
        .sig
        .inputs
        .iter()
        .filter(|arg| match arg {
            FnArg::Typed(pat_typed) => {
                if let Pat::Ident(pat_ident) = &*pat_typed.pat {
                    pat_ident.ident != "stream"
                } else {
                    true
                }
            }
            _ => true,
        })
        .cloned()
        .collect::<Vec<_>>();

    input_fn.sig.inputs = Punctuated::from_iter(filtered_inputs);

    // Prepend default stream initialization
    let default_stream_stmt = match input.map(|input| input.device) {
        Some(DeviceType::Cpu) => parse_quote! {
            let stream = StreamOrDevice::cpu();
        },
        Some(DeviceType::Gpu) => parse_quote! {
            let stream = StreamOrDevice::gpu();
        },
        None => parse_quote! {
            let stream = StreamOrDevice::default();
        },
    };
    input_fn.block.stmts.insert(0, default_stream_stmt);

    // Combine the original and modified functions into the output
    let expanded = quote! {
        #original_fn

        #input_fn
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(GenerateDtypeTestCases)]
pub fn generate_test_cases(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let tests = quote! {
        /// MLX's rules for promoting two dtypes.
        #[rustfmt::skip]
        const TYPE_RULES: [[Dtype; 13]; 13] = [
            // bool             uint8               uint16              uint32              uint64              int8                int16               int32               int64               float16             float32             bfloat16            complex64
            [Dtype::Bool,       Dtype::Uint8,       Dtype::Uint16,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int8,        Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Bfloat16,    Dtype::Complex64], // bool
            [Dtype::Uint8,      Dtype::Uint8,       Dtype::Uint16,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int16,       Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Bfloat16,    Dtype::Complex64], // uint8
            [Dtype::Uint16,     Dtype::Uint16,      Dtype::Uint16,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int32,       Dtype::Int32,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Bfloat16,    Dtype::Complex64], // uint16
            [Dtype::Uint32,     Dtype::Uint32,      Dtype::Uint32,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Bfloat16,    Dtype::Complex64], // uint32
            [Dtype::Uint64,     Dtype::Uint64,      Dtype::Uint64,      Dtype::Uint64,      Dtype::Uint64,      Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float16,     Dtype::Float32,     Dtype::Bfloat16,    Dtype::Complex64], // uint64
            [Dtype::Int8,       Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float32,     Dtype::Int8,        Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Bfloat16,    Dtype::Complex64], // int8
            [Dtype::Int16,      Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float32,     Dtype::Int16,       Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Bfloat16,    Dtype::Complex64], // int16
            [Dtype::Int32,      Dtype::Int32,       Dtype::Int32,       Dtype::Int64,       Dtype::Float32,     Dtype::Int32,       Dtype::Int32,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Bfloat16,    Dtype::Complex64], // int32
            [Dtype::Int64,      Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Float32,     Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Bfloat16,    Dtype::Complex64], // int64
            [Dtype::Float16,    Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float32,     Dtype::Float32,     Dtype::Complex64], // float16
            [Dtype::Float32,    Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Complex64], // float32
            [Dtype::Bfloat16,   Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Float32,     Dtype::Float32,     Dtype::Bfloat16,    Dtype::Complex64], // bfloat16
            [Dtype::Complex64,  Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64], // complex64
        ];

        #[cfg(test)]
        mod generated_tests {
            use super::*;
            use strum::IntoEnumIterator;
            use pretty_assertions::assert_eq;

            #[test]
            fn test_all_combinations() {
                for a in #name::iter() {
                    for b in #name::iter() {
                        let result = a.promote_with(b);
                        let expected = TYPE_RULES[a as usize][b as usize];
                        assert_eq!(result, expected, "{}", format!("Failed promotion test for {:?} and {:?}", a, b));
                    }
                }
            }
        }
    };

    TokenStream::from(tests)
}

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

#[proc_macro_attribute]
pub fn option_builder(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let builder = option_builder::expand_option_builder(&input).unwrap();
    TokenStream::from(builder)
}
