use std::default;

use darling::FromDeriveInput;
use quote::{quote, ToTokens};
use syn::{DeriveInput, Ident, ImplGenerics, TypeGenerics, WhereClause};

use crate::shared::{MandatoryField, OptionalField, PathOrIdent, Result};

#[derive(Debug, Clone, FromDeriveInput)]
#[darling(attributes(builder))]
pub(crate) struct BuilderStructProperty {
    pub ident: Ident,
    
    #[darling(default)]
    pub manual_impl: bool,
    
    pub root: Option<syn::Path>,
}

#[derive(Debug, darling::FromField, PartialEq)]
#[darling(attributes(builder))]
pub(crate) struct BuilderFieldProperty {
    pub ident: Option<syn::Ident>,
    
    pub ty: syn::Type,
    
    #[darling(default)]
    pub optional: bool,
    
    pub default: Option<syn::Path>,
    
    pub rename: Option<String>,
    
    #[darling(default)]
    pub ignore: bool,

    pub ty_override: Option<syn::Path>,
}

pub(crate) fn expand_derive_builder(input: DeriveInput) -> Result<proc_macro2::TokenStream> {
    todo!()
}

fn impl_builder_setters(
    builder_struct_ident: &PathOrIdent,
    optional_fields: &[OptionalField],
) -> proc_macro2::TokenStream {
    let setters = optional_fields.iter().map(|field| {
        let ident = &field.ident;
        let ty = &field.ty;
        let default = &field.default;
        quote! {
            pub fn #ident(mut self, #ident: impl Into<Option<#ty>>) -> Self {
                self.#ident = #ident.into().unwrap_or(#default);
                self
            }
        }
    });

    quote! {
        impl #builder_struct_ident {
            #(#setters)*
        }
    }
}

fn impl_builder_new(
    builder_struct_ident: &PathOrIdent,
    mandatory_fields: &[MandatoryField],
    optional_fields: &[OptionalField],
) -> proc_macro2::TokenStream {
    let mandatory_field_idents = mandatory_fields
        .iter()
        .map(|field| &field.ident)
        .collect::<Vec<_>>();
    let mandatory_field_types = mandatory_fields.iter().map(|field| &field.ty);

    let optional_field_idents = optional_fields.iter().map(|field| &field.ident);
    let optional_field_defaults = optional_fields.iter().map(|field| &field.default);

    quote! {
        impl #builder_struct_ident {
            pub fn new(#(#mandatory_field_idents: #mandatory_field_types),*) -> Self {
                Self {
                    #(#mandatory_field_idents,)*
                    #(#optional_field_idents: #optional_field_defaults,)*
                }
            }
        }
    }
}

fn impl_builder_trait<'a>(
    builder_struct_ident: &PathOrIdent,
    struct_ident: &Ident,
    root: &syn::Path,
    impl_generics: &ImplGenerics<'a>,
    type_generics: &TypeGenerics<'a>,
    where_clause: Option<&'a WhereClause>,
    mandatory_fields: &[MandatoryField],
    optional_fields: &[OptionalField],
) -> proc_macro2::TokenStream {
    let mandatory_field_idents = mandatory_fields.iter().map(|field| &field.ident);
    let optional_field_idents = optional_fields.iter().map(|field| &field.ident);

    quote! {
        impl #impl_generics #root::builder::Builder<#struct_ident> for #builder_struct_ident #type_generics #where_clause {
            type Error = std::convert::Infallible;

            fn build(self) -> std::result::Result<#struct_ident, Self::Error> {
                Ok(#struct_ident {
                    #(#mandatory_field_idents: self.#mandatory_field_idents,)*
                    #(#optional_field_idents: self.#optional_field_idents,)*
                })
            }
        }
    }
}

pub(crate) fn impl_builder<'a>(
    builder_struct_ident: &PathOrIdent,
    struct_ident: &Ident,
    root: &syn::Path,
    impl_generics: &ImplGenerics<'a>,
    type_generics: &TypeGenerics<'a>,
    where_clause: Option<&'a WhereClause>,
    mandatory_fields: &[MandatoryField],
    optional_fields: &[OptionalField],
    manual_impl: bool,
) -> proc_macro2::TokenStream {
    let builder_new = impl_builder_new(builder_struct_ident, mandatory_fields, optional_fields);
    let builder_setters = impl_builder_setters(builder_struct_ident, optional_fields);
    let builder_trait = if !manual_impl {
        impl_builder_trait(
            builder_struct_ident,
            struct_ident,
            root,
            impl_generics,
            type_generics,
            where_clause,
            mandatory_fields,
            optional_fields,
        )
    } else {
        quote! {}
    };

    quote! {
        #builder_new
        #builder_setters
        #builder_trait
    }
}
