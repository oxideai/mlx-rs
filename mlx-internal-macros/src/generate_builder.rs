use darling::FromDeriveInput;
use proc_macro2::TokenTree;
use quote::quote;
use syn::{DeriveInput, Ident, ImplGenerics, TypeGenerics, WhereClause};

use crate::{
    derive_buildable::StructProperty,
    shared::{
        parse_fields_from_derive_input, BuilderStructAnalyzer, BuilderStructProperty,
        MandatoryField, PathOrIdent, Result,
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
    let is_infallible = builder_struct_prop.err.is_none();
    let impl_struct_new = impl_struct_new(
        struct_ident,
        &root,
        &mandatory_fields,
        &impl_generics,
        &type_generics,
        where_clause,
        is_infallible,
    );

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

    Ok(quote! {
        #impl_struct_new
        #builder_struct
        #impl_builder
    })
}

fn impl_struct_new<'a>(
    struct_ident: &Ident,
    root: &syn::Path,
    mandatory_fields: &[MandatoryField],
    impl_generics: &'a ImplGenerics,
    type_generics: &TypeGenerics<'a>,
    where_clause: Option<&'a WhereClause>,
    is_infallible: bool,
) -> proc_macro2::TokenStream {
    let mandatory_field_idents = mandatory_fields
        .iter()
        .map(|field| &field.ident)
        .collect::<Vec<_>>();
    let mandatory_field_types = mandatory_fields.iter().map(|field| &field.ty);

    let doc = format!("Creates a new instance of `{}`.", struct_ident);

    // TODO: do we want to generate different code for infallible and fallible cases
    let ret = if is_infallible {
        quote! { -> Self }
    } else {
        quote! { -> std::result::Result<Self, <<Self as #root::builder::Buildable>::Builder as #root::builder::Builder<Self>>::Error> }
    };

    let unwrap_result = if is_infallible {
        quote! { .expect("Build with default parameters should not fail") }
    } else {
        quote! {}
    };

    quote! {
        impl #impl_generics #struct_ident #type_generics #where_clause {
            #[doc = #doc]
            pub fn new(#(#mandatory_field_idents: #mandatory_field_types),*) #ret
            {
                use #root::builder::Builder;
                <Self as #root::builder::Buildable>::Builder::new(#(#mandatory_field_idents),*).build()
                    #unwrap_result
            }
        }
    }
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
