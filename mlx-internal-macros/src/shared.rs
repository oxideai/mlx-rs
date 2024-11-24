use std::fmt::Display;

use quote::ToTokens;
use syn::{DeriveInput, Ident};
use darling::{FromDeriveInput, FromField};

pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, FromDeriveInput)]
#[darling(attributes(builder))]
pub(crate) struct BuilderStructProperty {
    pub ident: syn::Ident,
    
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

    pub setter: Option<syn::Ident>,
}

pub(crate) struct MandatoryField {
    pub ident: syn::Ident,
    pub ty: syn::Type,
}

pub(crate) struct OptionalField {
    pub ident: syn::Ident,
    pub ty: syn::Type,
    pub default: syn::Path,
    pub setter: Option<Ident>,
}


pub(crate) fn parse_fields_from_derive_input(
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

    let field_props = fields.iter().map(BuilderFieldProperty::from_field);

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

            optional_fields.push(OptionalField { ident, ty, default, setter: field_prop.setter });
        } else {
            mandatory_fields.push(MandatoryField { ident, ty });
        }
    }

    Ok((mandatory_fields, optional_fields))
}


#[derive(Debug, Clone)]
pub(crate) enum PathOrIdent {
    Path(syn::Path),
    Ident(syn::Ident),
}

impl ToTokens for PathOrIdent {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            PathOrIdent::Path(path) => path.to_tokens(tokens),
            PathOrIdent::Ident(ident) => ident.to_tokens(tokens),
        }
    }
}

impl Display for PathOrIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PathOrIdent::Path(path) => path.to_token_stream().fmt(f),
            PathOrIdent::Ident(ident) => Display::fmt(ident, f),
        }
    }
}