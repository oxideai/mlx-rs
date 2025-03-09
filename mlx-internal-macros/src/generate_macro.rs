use darling::FromMeta;
use itertools::Itertools;
use proc_macro::TokenStream;
use quote::quote;
use syn::{FnArg, Ident, ItemFn, Meta};

#[derive(Default, Debug, FromMeta)]
#[darling(default)]
struct Customize {
    root: Option<syn::LitStr>,
    default_dtype: Option<syn::Path>,
}

fn contains_optional_attribute(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("optional") {
            return true;
        }
    }
    false
}

fn remove_optional_attribute(attrs: &mut Vec<syn::Attribute>) {
    attrs.retain(|attr| {
        if attr.path().is_ident("optional") {
            return false;
        }
        true
    });
}

fn optional_arg_inputs(
    optional_arg_idents: &[Ident],
    optional_arg_mask: &[bool],
) -> Vec<proc_macro2::TokenStream> {
    optional_arg_idents
        .iter()
        .zip(optional_arg_mask.iter())
        .map(|(ident, mask)| {
            if *mask {
                quote! { $#ident }
            } else {
                quote! { None }
            }
        })
        .collect()
}

pub fn expand_generate_macro(
    attr: Option<Meta>,
    mut item: ItemFn, // The original function should be kept as is
) -> Result<TokenStream, syn::Error> {
    let customize = match attr {
        Some(attr) => Customize::from_meta(&attr).map_err(|e| syn::Error::new_spanned(attr, e))?,
        None => Customize::default(),
    };

    // The mod path where the function can be accessed publicly
    let fn_mod_path = match customize.root {
        Some(lit_str) => {
            let tokens: proc_macro2::TokenStream = lit_str.parse()?;
            quote! { #tokens }
        },
        None => quote! { $crate::ops },
    };

    let (default_generics, dtype_generics) = handle_generic_args(&item.sig.generics, &customize.default_dtype);

    let args = item
        .sig
        .inputs
        .iter_mut()
        .map(|arg| match arg {
            FnArg::Receiver(_) => Err(syn::Error::new_spanned(arg, "self is not allowed")),
            FnArg::Typed(pat_type) => Ok(pat_type),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut mandatory_arg_idents = Vec::new();
    let mut optional_arg_idents = Vec::new();
    for arg in args {
        match &*arg.pat {
            syn::Pat::Ident(ident) => {
                if contains_optional_attribute(&arg.attrs) {
                    optional_arg_idents.push(ident.ident.clone());
                } else {
                    if !optional_arg_idents.is_empty() {
                        return Err(syn::Error::new_spanned(
                            &arg.pat,
                            "mandatory arguments must precede optional arguments",
                        ));
                    }
                    mandatory_arg_idents.push(ident.ident.clone());
                }
            }
            _ => return Err(syn::Error::new_spanned(&arg.pat, "unsupported pattern")),
        }
        remove_optional_attribute(&mut arg.attrs);
    }

    // Check if the last optional argument is `stream`
    if let Some(last_optional_arg) = optional_arg_idents.last() {
        if last_optional_arg != "stream" {
            return Err(syn::Error::new_spanned(
                &item,
                "the last optional argument must be `stream`",
            ));
        }
    }
    // Remove the last optional argument `stream`
    optional_arg_idents.pop();

    // Remove "_device" suffix from the macro name if it exists
    let fn_ident = &item.sig.ident;
    
    let generated = generate_macro(
        fn_mod_path,
        fn_ident,
        &mandatory_arg_idents,
        &optional_arg_idents,
        default_generics,
        dtype_generics,
    )?;

    let output = quote! {
        #item
        #generated
    };

    Ok(output.into())
}

/// If there are generic arguments, the last argument is assumed to be `dtype`.
/// 
/// Returns two `syn::Generics`:
/// 1. With the last argument set to `f32`
/// 2. With the last argument set to `$dtype`
fn handle_generic_args(generic_args: &syn::Generics, default_dtype: &Option<syn::Path>) -> (proc_macro2::TokenStream, Option<proc_macro2::TokenStream>) {
    // Count number of generic type arguments
    let count = generic_args
        .params
        .iter()
        .filter(|param| matches!(param, syn::GenericParam::Type(_)))
        .count();

    if count == 0 {
        return (quote!{}, None)
    }
    
    // All generics arguments except for the last one will be inferred
    let infer_tokens = vec![quote! { _ }; count - 1];
    
    let default_generics = match default_dtype {
        Some(path) => quote! { ::<#(#infer_tokens,)* #path> },
        None => quote! { ::<#(#infer_tokens,)* f32> },
    };
    let dtype_generics = quote! { ::<#(#infer_tokens,)* $dtype> };

    (default_generics, Some(dtype_generics))
}

fn generate_macro(
    fn_mod_path: proc_macro2::TokenStream,
    fn_ident: &Ident,
    mandatory_arg_idents: &[Ident],
    optional_arg_idents: &[Ident],
    default_generics: proc_macro2::TokenStream,
    dtype_generics: Option<proc_macro2::TokenStream>,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let mut trimmed_fn_ident_str = fn_ident.to_string();
    if trimmed_fn_ident_str.ends_with("_device") {
        trimmed_fn_ident_str = trimmed_fn_ident_str.trim_end_matches("_device").to_string();
    }
    let trimmed_fn_ident = Ident::new(&trimmed_fn_ident_str, fn_ident.span());

    let mut macro_variants = Vec::new();
    generate_macro_variants(
        &fn_mod_path,
        fn_ident,
        &trimmed_fn_ident,
        mandatory_arg_idents,
        optional_arg_idents,
        default_generics,
        dtype_generics,
        &mut macro_variants,
    );

    let macro_docs = format!(
        "Macro generated for the function `{}::{}`. See the function documentation for more details.",
        fn_mod_path, fn_ident
    );

    let generated = quote! {
        #[doc = #macro_docs]
        #[macro_export]
        macro_rules! #trimmed_fn_ident {
            #(
                #macro_variants
            )*
        }
    };

    Ok(generated)
}

fn generate_macro_variants(
    fn_mod_path: &proc_macro2::TokenStream,
    fn_ident: &Ident,
    trimmed_fn_ident: &Ident,
    mandatory_arg_idents: &[Ident],
    optional_arg_idents: &[Ident],
    default_generics: proc_macro2::TokenStream,
    dtype_generics: Option<proc_macro2::TokenStream>,
    macro_variants: &mut Vec<proc_macro2::TokenStream>,
) {
    // A mask of bool indicating whether the optional argument is selected
    let mut optional_arg_mask = vec![false; optional_arg_idents.len()];

    let optional_arg_input = optional_arg_inputs(&optional_arg_idents, &optional_arg_mask);
    let mandatory_only_variant_body = quote! {
        (
            #($#mandatory_arg_idents: expr),*
        ) => {
            #fn_mod_path::#trimmed_fn_ident #default_generics(#($#mandatory_arg_idents),*, #(#optional_arg_input),*)
        };
        (
            #($#mandatory_arg_idents: expr),*,
            stream=$stream:expr
        ) => {
            #fn_mod_path::#fn_ident #default_generics(#($#mandatory_arg_idents),*, #(#optional_arg_input,)* $stream)
        };
    };
    macro_variants.push(mandatory_only_variant_body);

    if let Some(dtype_generics) = &dtype_generics {
        let mandatory_only_variant_body = quote! {
            (
                #($#mandatory_arg_idents: expr),*,
                dtype=$dtype:ty
            ) => {
                #fn_mod_path::#trimmed_fn_ident #dtype_generics(#($#mandatory_arg_idents),*, #(#optional_arg_input),*)
            };
            (
                #($#mandatory_arg_idents: expr),*,
                dtype=$dtype:ty,
                stream=$stream:expr
            ) => {
                #fn_mod_path::#fn_ident #dtype_generics(#($#mandatory_arg_idents),*, #(#optional_arg_input,)* $stream)
            };
        };
        macro_variants.push(mandatory_only_variant_body);
    }

    for perms in 1..optional_arg_idents.len() + 1 {
        let permuted_indices = (0..optional_arg_idents.len()).permutations(perms);
        for indices in permuted_indices {
            indices.iter().for_each(|&i| optional_arg_mask[i] = true);

            let selected_optional_arg_idents: Vec<_> =
                indices.iter().map(|&i| &optional_arg_idents[i]).collect();

            let optional_arg_input = optional_arg_inputs(&optional_arg_idents, &optional_arg_mask);
            let macro_variant_body = quote! {
                (
                    #($#mandatory_arg_idents: expr),*,
                    #(
                        #selected_optional_arg_idents=$#selected_optional_arg_idents:expr
                    ),*
                ) => {
                    #fn_mod_path::#trimmed_fn_ident #default_generics(#($#mandatory_arg_idents),*, #(#optional_arg_input),*)
                };
                (
                    #($#mandatory_arg_idents: expr),*,
                    #(
                        #selected_optional_arg_idents=$#selected_optional_arg_idents:expr
                    ),*,
                    stream=$stream:expr
                ) => {
                    #fn_mod_path::#fn_ident #default_generics(#($#mandatory_arg_idents),*, #(#optional_arg_input,)* $stream)
                };
            };

            macro_variants.push(macro_variant_body);

            if let Some(dtype_generics) = &dtype_generics {
                let macro_variant_body = quote! {
                    (
                        #($#mandatory_arg_idents: expr),*,
                        #(
                            #selected_optional_arg_idents=$#selected_optional_arg_idents:expr
                        ),*,
                        dtype=$dtype:ty
                    ) => {
                        #fn_mod_path::#trimmed_fn_ident #dtype_generics(#($#mandatory_arg_idents),*, #(#optional_arg_input),*)
                    };
                    (
                        #($#mandatory_arg_idents: expr),*,
                        #(
                            #selected_optional_arg_idents=$#selected_optional_arg_idents:expr
                        ),*,
                        dtype=$dtype:ty,
                        stream=$stream:expr
                    ) => {
                        #fn_mod_path::#fn_ident #dtype_generics(#($#mandatory_arg_idents),*, #(#optional_arg_input,)* $stream)
                    };
                };

                macro_variants.push(macro_variant_body);
            }

            // Reset the mask
            optional_arg_mask.iter_mut().for_each(|b| *b = false);
        }
    }
}
