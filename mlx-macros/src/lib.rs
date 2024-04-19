extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, ItemFn, FnArg, Pat, parse_quote};
use syn::punctuated::Punctuated;

#[proc_macro_attribute]
pub fn default_device(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let original_fn = input_fn.clone();

    // Ensure function name convention
    if !input_fn.sig.ident.to_string().ends_with("_device") {
        panic!("Function name must end with '_device'");
    }
    let new_fn_name = format_ident!("{}", &input_fn.sig.ident.to_string().trim_end_matches("_device"));
    input_fn.sig.ident = new_fn_name;

    // Filter out the `stream` parameter and reconstruct the Punctuated collection
    let filtered_inputs = input_fn.sig.inputs.iter().filter(|arg| {
        match arg {
            FnArg::Typed(pat_typed) => {
                if let Pat::Ident(pat_ident) = &*pat_typed.pat {
                    pat_ident.ident != "stream"
                } else {
                    true
                }
            },
            _ => true
        }
    }).cloned().collect::<Vec<_>>();

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
