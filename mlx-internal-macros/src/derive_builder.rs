
use darling::FromDeriveInput;
use quote::quote;
use syn::{DeriveInput, Ident, ImplGenerics, TypeGenerics, WhereClause};

use crate::shared::{parse_fields_from_derive_input, MandatoryField, OptionalField, PathOrIdent, Result, BuilderStructProperty};


pub(crate) fn expand_derive_builder(input: DeriveInput) -> Result<proc_macro2::TokenStream> {
    let struct_prop = BuilderStructProperty::from_derive_input(&input)?;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();

    let builder_ident = &struct_prop.ident;
    if !is_builder_struct_end_with_builder(builder_ident) {
        return Err("Builder struct must end with 'Builder'".into());
    }
    let builder_ident_str = builder_ident.to_string();
    let struct_ident = Ident::new(
        // We have already checked that the builder struct ends with 'Builder'
        &builder_ident_str[..builder_ident_str.len() - "Builder".len()],
        builder_ident.span(),
    );
    let root = match struct_prop.root {
        Some(path) => path,
        None => syn::parse_quote!(::mlx_rs),
    };

    let builder_struct_ident = PathOrIdent::Ident(builder_ident.clone());
    let (mandatory_fields, optional_fields) = parse_fields_from_derive_input(&input)?;
    Ok(impl_builder(
        &builder_struct_ident,
        &struct_ident,
        &root,
        &impl_generics,
        &type_generics,
        where_clause,
        &mandatory_fields,
        &optional_fields,
        struct_prop.manual_impl,
    ))
}

fn is_builder_struct_end_with_builder(ident: &Ident) -> bool {
    ident.to_string().ends_with("Builder")
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

#[allow(clippy::too_many_arguments)]
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
