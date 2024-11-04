extern crate proc_macro;
use darling::FromMeta;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, parse_quote, FnArg, ItemEnum, ItemFn, ItemStruct, Pat};

mod generate_builder;

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

#[doc(hidden)]
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

#[doc(hidden)]
#[proc_macro]
pub fn generate_test_cases(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemEnum);
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

    TokenStream::from(quote! {
        #input
        #tests
    })
}

/// This is for internal use only
///
/// The struct that this macro is applied to should NOT derive the `Default` trait.
///
/// # Struct Attribute(s)
///
/// This macro takes the following attributes:
///
/// - `generate_builder`: This attribute should be applied on the struct.
///
///   Arguments
///
///   - `generate_build_fn = <bool>`: It defaults to `true` if not specified. If `true`, it will:
///     1. generate a `<Struct>Builder::build` function that takes the mandatory fields as arguments
///        and returns the struct.
///     2. generate a `<Struct>::new` function that takes the mandatory fields as arguments and
///        returns the struct. This is a convenience function that simply calls
///        `<Struct>Builder::new().build(...)`. Additionally, if there is NO mandatory field, it
///        will implement the `Default` trait for the struct.
///
/// # Field Attribute(s)
///
/// - `optional`: This attribute should be applied on the field. It indicates that the field is
///   optional. Behaviour of the generated builder struct depends on the argument of this attribute.
///   
///   Arguments
///   
///   - `skip = <bool>`: Default `false`. If `true`, the macro will NOT generate a setter for this
///     field in the builder struct. It will also NOT wrap the field in an `Option` in the struct,
///     and this field will remain as its original type in the builder struct. It is the user's
///     responsibility to implement the setter for this field in the builder struct.
///
///     The `build` function cannot be generated if any field is marked as `skip = true`, and an
///     error will be shown in that case.
///
///   - `ty = <Path>`: If set, the optional field in the builder will be of the type specified by
///     this argument. This is useful when the field is optional and the type is not the same as the
///     original field type.
///
///   - `default_value = <Path>`: This argument is required if a default build function were to be
///     generated. It specifies the default value for the field. The value should be a `Path`
///     (something that is interpreted as a `syn::Path`) to a constant or an enum variant.
///
/// # Generate Build Function
///
/// The following conditions have to be met to generate the `<Type>Builder::build` function and
/// `<Type>::new` function:
///
///   1. `generate_build_fn = true` in the `generate_builder` attribute.
///   2. No optional field is marked as `skip = true`.
///   3. No optional field is marked as `ty = <Path>`.
///
/// Otherwise, the user must implement the `<Type>Builder::build` function and `<Type>::new`
/// manually.
#[doc(hidden)]
#[proc_macro]
pub fn generate_builder(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemStruct);
    let builder = generate_builder::expand_generate_builder(input).unwrap();
    TokenStream::from(builder)
}
