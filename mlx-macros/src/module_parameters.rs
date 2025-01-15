use darling::{FromDeriveInput, FromField};
use syn::{DataStruct, DeriveInput, Generics, Ident};

use crate::util::filter_fields_with_attr;

#[derive(Debug, Clone, FromDeriveInput)]
#[darling(attributes(module))]
struct ModuleProperties {
    root: Option<syn::Path>,
}

#[derive(Debug, Clone, FromField)]
#[darling(attributes(param))]
struct FieldParameters {
    rename: Option<String>,
}

pub(crate) fn expand_module_parameters(
    input: &DeriveInput,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let prop = ModuleProperties::from_derive_input(input)?;
    let struct_ident = &input.ident;
    let generics = &input.generics;
    match &input.data {
        syn::Data::Struct(data) => {
            expand_module_parameters_for_struct(struct_ident, generics, data, prop.root)
        }
        _ => Err(syn::Error::new_spanned(
            input,
            "ModuleParameters can only be derived for structs",
        )),
    }
}

fn expand_module_parameters_for_struct(
    ident: &Ident,
    generics: &Generics,
    data: &DataStruct,
    root: Option<syn::Path>,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let fields = filter_fields_with_attr(&data.fields, "param")?;

    Ok(impl_module_parameters_for_struct(
        ident,
        generics,
        fields.filtered,
        root,
    ))
}

fn impl_module_parameters_for_struct(
    ident: &Ident,
    generics: &Generics,
    fields: Vec<&syn::Field>,
    root: Option<syn::Path>,
) -> proc_macro2::TokenStream {
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let field_idents: Vec<_> = fields.iter().map(|field| field.ident.as_ref().unwrap()).collect();
    let field_names: Vec<_> = fields.iter()
        .map(|field| {
            let field_params = FieldParameters::from_field(field).unwrap();
            match field_params.rename {
                Some(rename) => {
                    let span = field.ident.as_ref().expect("Expect named field").span();
                    syn::Ident::new(&rename, span)
                },
                None => field.ident.as_ref().unwrap().clone(),
            }
        })
        .collect();

    // Returns None if there are no fields
    let default_all_frozen = match field_names.len() {
        0 => quote::quote! { None },
        _ => quote::quote! { Some(true) },
    };

    // Returns None if there are no fields
    let default_any_frozen = match field_names.len() {
        0 => quote::quote! { None },
        _ => quote::quote! { Some(false) },
    };

    let (extern_import, root) = match root {
        Some(root) => (quote::quote! {}, quote::quote! { #root }),
        None => (
            quote::quote! { extern crate mlx_rs as _mlx_rs; },
            quote::quote! { _mlx_rs },
        ),
    };

    quote::quote! {
        const _: () = {
            #extern_import
            impl #impl_generics #root::module::ModuleParameters for #ident #ty_generics #where_clause {
                fn freeze_parameters(&mut self, recursive: bool) {
                    use #root::module::Parameter;
                    #(self.#field_idents.freeze(recursive);)*
                }

                fn unfreeze_parameters(&mut self, recursive: bool) {
                    use #root::module::Parameter;
                    #(self.#field_idents.unfreeze(recursive);)*
                }

                fn parameters(&self) -> #root::module::ModuleParamRef<'_> {
                    let mut parameters = #root::nested::NestedHashMap::new();
                    #(parameters.insert(std::rc::Rc::from(stringify!(#field_names)), #root::module::Parameter::as_nested_value(&self.#field_idents));)*
                    parameters
                }

                fn parameters_mut(&mut self) -> #root::module::ModuleParamMut<'_> {
                    let mut parameters = #root::nested::NestedHashMap::new();
                    #(parameters.insert(std::rc::Rc::from(stringify!(#field_names)), #root::module::Parameter::as_nested_value_mut(&mut self.#field_idents));)*
                    parameters
                }

                fn trainable_parameters(&self) -> #root::module::ModuleParamRef<'_> {
                    let mut parameters = #root::nested::NestedHashMap::new();
                    #(
                        if let Some(field) = #root::module::Parameter::as_trainable_nested_value(&self.#field_idents) {
                            parameters.insert(std::rc::Rc::from(stringify!(#field_names)), field);
                        }
                    )*
                    parameters
                }

                fn all_frozen(&self) -> Option<bool> {
                    use #root::module::Parameter;
                    #(
                        if matches!(self.#field_idents.is_frozen(), Some(false)) {
                            return Some(false);
                        }
                    )*
                    #default_all_frozen
                }

                fn any_frozen(&self) -> Option<bool> {
                    use #root::module::Parameter;
                    #(
                        if matches!(self.#field_idents.is_frozen(), Some(true)) {
                            return Some(true);
                        }
                    )*
                    #default_any_frozen
                }
            }
        };
    }
}
