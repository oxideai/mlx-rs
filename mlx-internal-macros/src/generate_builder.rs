use darling::FromDeriveInput;
use quote::quote;
use syn::{DeriveInput, Ident, ImplGenerics, TypeGenerics, WhereClause};

use crate::{
    derive_buildable::StructProperty,
    derive_builder::impl_builder,
    shared::{
        parse_fields_from_derive_input, BuilderStructProperty,
        MandatoryField, OptionalField, PathOrIdent, Result,
    },
};

pub(crate) fn expand_generate_builder(input: &DeriveInput) -> Result<proc_macro2::TokenStream> {
    let struct_prop = StructProperty::from_derive_input(input)?;
    let manual_impl_builder_trait = BuilderStructProperty::from_derive_input(input)?.manual_impl;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();

    let struct_ident = &struct_prop.ident;
    let builder_ident = syn::Ident::new(&format!("{}Builder", struct_ident), struct_ident.span());
    let root = match struct_prop.root {
        Some(path) => path,
        None => syn::parse_quote!(::mlx_rs),
    };

    let struct_builder_ident = match &struct_prop.builder {
        Some(path) => PathOrIdent::Path(path.clone()),
        None => PathOrIdent::Ident(builder_ident),
    };

    let (mandatory_fields, optional_fields) = parse_fields_from_derive_input(input)?;
    let impl_struct_new = impl_struct_new(
        struct_ident,
        &root,
        &mandatory_fields,
        &impl_generics,
        &type_generics,
        where_clause,
    );

    let builder_struct = if struct_prop.builder.is_none() {
        generate_builder_struct(
            &struct_ident,
            &struct_builder_ident,
            &mandatory_fields,
            &optional_fields,
            &type_generics,
            where_clause,
        )
    } else {
        quote! {}
    };
    let impl_builder = impl_builder(
        &struct_builder_ident,
        struct_ident,
        &root,
        &impl_generics,
        &type_generics,
        where_clause,
        &mandatory_fields,
        &optional_fields,
        manual_impl_builder_trait,
    );

    Ok(quote! {
        #impl_struct_new
        #builder_struct
        #impl_builder
    })
}

fn generate_builder_struct<'a>(
    struct_ident: &Ident,
    builder_ident: &PathOrIdent,
    mandatory_fields: &[MandatoryField],
    optional_fields: &[OptionalField],
    type_generics: &TypeGenerics<'a>,
    where_clause: Option<&'a WhereClause>,
) -> proc_macro2::TokenStream {
    let mandatory_field_idents = mandatory_fields.iter().map(|field| &field.ident);
    let mandatory_field_tys = mandatory_fields.iter().map(|field| &field.ty);

    let optional_field_idents = optional_fields.iter().map(|field| &field.ident);
    let optional_field_tys = optional_fields.iter().map(|field| &field.ty);

    let doc = format!("Builder for `{}`.", struct_ident);

    quote! {
        #[doc = #doc]
        pub struct #builder_ident #type_generics #where_clause {
            #(#mandatory_field_idents: #mandatory_field_tys,)*
            #(#optional_field_idents: #optional_field_tys,)*
        }
    }
}

fn impl_struct_new<'a>(
    struct_ident: &Ident,
    root: &syn::Path,
    mandatory_fields: &[MandatoryField],
    impl_generics: &'a ImplGenerics,
    type_generics: &TypeGenerics<'a>,
    where_clause: Option<&'a WhereClause>,
) -> proc_macro2::TokenStream {
    let mandatory_field_idents = mandatory_fields
        .iter()
        .map(|field| &field.ident)
        .collect::<Vec<_>>();
    let mandatory_field_types = mandatory_fields.iter().map(|field| &field.ty);

    let doc = format!("Creates a new instance of `{}`.", struct_ident);

    quote! {
        impl #impl_generics #struct_ident #type_generics #where_clause {
            #[doc = #doc]
            pub fn new(#(#mandatory_field_idents: #mandatory_field_types),*) 
                -> Result<Self, <<Self as #root::builder::Buildable>::Builder as #root::builder::Builder<Self>>::Error> 
            {
                <Self as #root::builder::Buildable>::Builder::new(#(#mandatory_field_idents),*).build()
            }
        }
    }
}
