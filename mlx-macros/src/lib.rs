extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, parse_quote, DeriveInput, FnArg, ItemFn, Pat};

#[proc_macro_attribute]
pub fn default_device(_attr: TokenStream, item: TokenStream) -> TokenStream {
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
    let default_stream_stmt = parse_quote! {
        let stream = StreamOrDevice::default();
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
