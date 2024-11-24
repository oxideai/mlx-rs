use darling::{FromDeriveInput, FromField};
use quote::quote;
use syn::{DeriveInput, TypeGenerics, WhereClause};

use crate::{
    derive_buildable::StructProperty,
    derive_builder::{self, impl_builder, BuilderFieldProperty, BuilderStructProperty},
    shared::{MandatoryField, OptionalField, PathOrIdent, Result},
};

pub(crate) fn expand_generate_builder(input: &DeriveInput) -> Result<proc_macro2::TokenStream> {
    let struct_prop = StructProperty::from_derive_input(&input)?;
    let manual_impl_builder_trait = BuilderStructProperty::from_derive_input(&input)?
        .manual_impl;
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

    let (mandatory_fields, optional_fields) = parse_fields_from_derive_input(&input)?;
    let builder_struct = if struct_prop.builder.is_none() {
        generate_builder_struct(
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
        #builder_struct
        #impl_builder
    })
}

fn parse_fields_from_derive_input(
    item: &DeriveInput,
) -> Result<(Vec<MandatoryField>, Vec<OptionalField>)> {
    match &item.data {
        syn::Data::Struct(data) => parse_fields_from_datastruct(data),
        _ => Err("Only structs are supported".into()),
    }
}

fn parse_fields_from_datastruct(
    item: &syn::DataStruct,
) -> Result<(Vec<MandatoryField>, Vec<OptionalField>)> {
    parse_fields(&item.fields)
}

fn parse_fields(fields: &syn::Fields) -> Result<(Vec<MandatoryField>, Vec<OptionalField>)> {
    let mut mandatory_fields = Vec::new();
    let mut optional_fields = Vec::new();

    let field_props = fields.iter().map(|field| BuilderFieldProperty::from_field(field));

    for field_prop in field_props {
        let field_prop = field_prop?;
        if field_prop.ignore {
            continue;
        }

        let mut ident = match field_prop.ident {
            Some(ident) => ident,
            None => return Err("Unnamed fields are not supported".into()),
        };

        if let Some(rename) = field_prop.rename {
            ident = syn::Ident::new(&rename, ident.span());
        }

        let ty = match field_prop.ty_override {
            Some(ty_override) => syn::Type::Path(syn::TypePath {
                qself: None,
                path: ty_override,
            }),
            None => field_prop.ty,
        };

        if field_prop.optional {
            let default = match field_prop.default {
                Some(default) => default,
                None => {
                    return Err(
                        format!("Field {} is optional but has no default value", ident).into(),
                    )
                }
            };

            optional_fields.push(OptionalField { ident, ty, default });
        } else {
            mandatory_fields.push(MandatoryField { ident, ty });
        }
    }

    Ok((mandatory_fields, optional_fields))
}

fn generate_builder_struct<'a>(
    ident: &PathOrIdent,
    mandatory_fields: &[MandatoryField],
    optional_fields: &[OptionalField],
    type_generics: &TypeGenerics<'a>,
    where_clause: Option<&'a WhereClause>,
) -> proc_macro2::TokenStream {
    let mandatory_field_idents = mandatory_fields.iter().map(|field| &field.ident);
    let mandatory_field_tys = mandatory_fields.iter().map(|field| &field.ty);

    let optional_field_idents = optional_fields.iter().map(|field| &field.ident);
    let optional_field_tys = optional_fields.iter().map(|field| &field.ty);

    quote! {
        struct #ident #type_generics #where_clause {
            #(#mandatory_field_idents: #mandatory_field_tys,)*
            #(#optional_field_idents: #optional_field_tys,)*
        }
    }
}
