extern crate proc_macro;
use darling::FromMeta;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, parse_quote, FnArg, ItemFn, Pat};

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
