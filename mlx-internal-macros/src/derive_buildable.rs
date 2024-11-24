use darling::{FromDeriveInput, FromField, FromMeta};
use quote::{quote, ToTokens};
use syn::{DeriveInput, ItemStruct, TypeGenerics, WhereClause};

use crate::{derive_builder::impl_builder, shared::{MandatoryField, OptionalField, PathOrIdent, Result}};

#[derive(Debug, Clone, FromDeriveInput)]
#[darling(attributes(buildable))]
#[allow(dead_code)]
pub(crate) struct StructProperty {
    pub ident: syn::Ident,
    
    /// Generate builder if None
    pub builder: Option<syn::Path>,

    /// Rename `mlx_rs` if Some(_)
    pub root: Option<syn::Path>,
}

pub(crate) fn expand_derive_buildable(input: DeriveInput) -> Result<proc_macro2::TokenStream> {
    let struct_prop = StructProperty::from_derive_input(&input)?;
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

    let impl_buildable = quote! {
        impl #impl_generics #root::builder::Buildable for #struct_ident #type_generics #where_clause {
            type Builder = #struct_builder_ident;
        }
    };

    Ok(quote! {
        #impl_buildable
    })
}
