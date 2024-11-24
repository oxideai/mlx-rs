use darling::FromDeriveInput;
use syn::{DeriveInput, Ident};
use quote::quote;

use crate::shared::{
    parse_fields_from_derive_input, BuilderStructAnalyzer, BuilderStructProperty, PathOrIdent,
    Result,
};

pub(crate) fn expand_derive_builder(input: DeriveInput) -> Result<proc_macro2::TokenStream> {
    let builder_struct_prop = BuilderStructProperty::from_derive_input(&input)?;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();

    let builder_ident = &builder_struct_prop.ident;
    if !is_builder_struct_end_with_builder(builder_ident) {
        return Err("Builder struct must end with 'Builder'".into());
    }
    let builder_ident_str = builder_ident.to_string();
    let struct_ident = Ident::new(
        // We have already checked that the builder struct ends with 'Builder'
        &builder_ident_str[..builder_ident_str.len() - "Builder".len()],
        builder_ident.span(),
    );
    let root = match builder_struct_prop.root {
        Some(path) => path,
        None => syn::parse_quote!(::mlx_rs),
    };

    let builder_struct_ident = PathOrIdent::Ident(builder_ident.clone());
    let (mandatory_fields, optional_fields) = parse_fields_from_derive_input(&input)?;
    let is_default_infallible = builder_struct_prop.default_infallible.unwrap_or_else(|| builder_struct_prop.err.is_none());

    let builder_struct_analyzer = BuilderStructAnalyzer {
        struct_ident: &struct_ident,
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

    let impl_builder = builder_struct_analyzer.impl_builder();
    let impl_struct_new = builder_struct_analyzer.impl_struct_new(is_default_infallible);

    Ok(quote! {
        #impl_builder
        #impl_struct_new
    })
}

fn is_builder_struct_end_with_builder(ident: &Ident) -> bool {
    ident.to_string().ends_with("Builder")
}
