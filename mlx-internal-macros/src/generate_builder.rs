use darling::FromDeriveInput;
use proc_macro2::TokenTree;
use quote::quote;
use syn::DeriveInput;

use crate::{
    derive_buildable::StructProperty,
    shared::{
        parse_fields_from_derive_input, BuilderStructAnalyzer, BuilderStructProperty,
        PathOrIdent, Result,
    },
};

pub(crate) fn expand_generate_builder(input: &DeriveInput) -> Result<proc_macro2::TokenStream> {
    // Make sure the struct does NOT have #[derive(Default)]
    if struct_attr_derive_default(&input.attrs) {
        return Err("Struct with #[derive(Default)] cannot derive Buildable".into());
    }

    let struct_prop = StructProperty::from_derive_input(input)?;
    let builder_struct_prop = BuilderStructProperty::from_derive_input(input)?;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();

    let struct_ident = &struct_prop.ident;
    let builder_struct_ident =
        syn::Ident::new(&format!("{}Builder", struct_ident), struct_ident.span());
    let root = match struct_prop.root {
        Some(path) => path,
        None => syn::parse_quote!(::mlx_rs),
    };

    let (mandatory_fields, optional_fields) = parse_fields_from_derive_input(input)?;
    let is_default_infallible = builder_struct_prop.default_infallible.unwrap_or_else(|| builder_struct_prop.err.is_none());

    let builder_struct_ident = match &struct_prop.builder {
        Some(path) => PathOrIdent::Path(path.clone()),
        None => PathOrIdent::Ident(builder_struct_ident.clone()),
    };
    let builder_struct_analyzer = BuilderStructAnalyzer {
        struct_ident,
        builder_struct_ident: &builder_struct_ident,
        root: &root,
        impl_generics: &impl_generics,
        type_generics: &type_generics,
        where_clause,
        mandatory_fields: &mandatory_fields,
        optional_fields: &optional_fields,
        build_with: builder_struct_prop.build_with.as_ref(),
        err: builder_struct_prop.err.as_ref(),
    };
    let builder_struct = if struct_prop.builder.is_none() {
        builder_struct_analyzer.generate_builder_struct()
    } else {
        quote! {}
    };
    let impl_builder = builder_struct_analyzer.impl_builder();
    let impl_struct_new = builder_struct_analyzer.impl_struct_new(is_default_infallible);

    Ok(quote! {
        #builder_struct
        #impl_builder
        #impl_struct_new
    })
}

fn struct_attr_derive_default(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("derive") {
                attr.meta
                    .require_list()
                    .map(|list| list.tokens.clone())
                    .ok()
            } else {
                None
            }
        })
        .any(|tokens| {
            tokens.into_iter().any(|tree| {
                if let TokenTree::Ident(ident) = tree {
                    ident == "Default"
                } else {
                    false
                }
            })
        })
}
