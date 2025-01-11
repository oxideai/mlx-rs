use darling::FromDeriveInput;
use syn::{DeriveInput, Generics, Ident};

use crate::util::{filter_fields_with_attr, FilteredFields};

#[derive(Debug, Clone, FromDeriveInput)]
#[darling(attributes(quantizable))]
struct StructProperties {
    root: Option<syn::Path>,
}

pub(crate) fn expand_quantizable(
    input: &DeriveInput,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let prop = StructProperties::from_derive_input(input)?;
    let struct_ident = &input.ident;
    let generics = &input.generics;

    match &input.data {
        syn::Data::Struct(data) => {
            expand_quantizable_module_for_struct(struct_ident, generics, data, prop.root)
        }
        _ => Err(syn::Error::new_spanned(
            input,
            "Quantizable can only be derived for structs",
        )),
    }
}

fn expand_quantizable_module_for_struct(
    ident: &syn::Ident,
    generics: &syn::Generics,
    data: &syn::DataStruct,
    root: Option<syn::Path>,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    // Filter fields with #[quantizable]
    let fields = filter_fields_with_attr(&data.fields, "quantizable")?;

    impl_quantizable_module_for_struct(ident, generics, fields, root)
}

fn impl_quantizable_module_for_struct(
    ident: &Ident,
    generics: &Generics,
    fields: FilteredFields,
    root: Option<syn::Path>,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    // let field_names: Vec<_> = fields.iter().map(|field| &field.ident).collect();

    let filtered_field_names = fields.filtered.iter().map(|field| &field.ident);
    let other_field_names = fields.other_fields.iter().map(|field| &field.ident);

    if fields.filtered.is_empty() {
        return Err(syn::Error::new_spanned(
            ident,
            "At least one field must be quantizable",
        ));
    }

    let (extern_import, root) = match root {
        Some(root) => (quote::quote! {}, quote::quote! { #root }),
        None => (
            quote::quote! { extern crate mlx_rs as _mlx_rs; },
            quote::quote! { _mlx_rs },
        ),
    };

    let token = quote::quote! {
        const _: () = {
            #extern_import
            impl #impl_generics #root::quantization::Quantizable for #ident #ty_generics #where_clause {
                type Quantized = Self; // Generating new struct is not supported yet

                type QuantizationError = #root::error::Exception;

                fn try_into_quantized(
                    self,
                    group_size: i32,
                    bits: i32,
                ) -> Result<Self::Quantized, Self::QuantizationError> {
                    Ok(Self {
                        #(
                            #filtered_field_names: #root::quantization::Quantizable
                                ::try_into_quantized(self.#filtered_field_names, group_size, bits)?,
                        )*
                        #(
                            #other_field_names: self.#other_field_names,
                        )*
                    })
                }
            }
        };
    };
    Ok(token)
}
