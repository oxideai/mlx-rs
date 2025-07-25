use darling::FromDeriveInput;
use syn::{DataStruct, DeriveInput, Generics, Ident};

use crate::util::filter_fields_with_attr;

#[derive(Debug, Clone, FromDeriveInput)]
#[darling(attributes(module))]
struct ModuleProperties {
    root: Option<syn::Path>,
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
    let field_names: Vec<_> = fields.iter().map(|field| &field.ident).collect();

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
                fn num_parameters(&self) -> usize {
                    use #root::module::Parameter;
                    let mut count = 0;
                    #(
                        count += self.#field_names.count();
                    )*
                    count
                }

                fn freeze_parameters(&mut self, recursive: bool) {
                    use #root::module::Parameter;
                    #(self.#field_names.freeze(recursive);)*
                }

                fn unfreeze_parameters(&mut self, recursive: bool) {
                    use #root::module::Parameter;
                    #(self.#field_names.unfreeze(recursive);)*
                }

                fn parameters(&self) -> #root::module::ModuleParamRef<'_> {
                    let mut parameters = #root::nested::NestedHashMap::new();
                    #(parameters.insert(std::rc::Rc::from(stringify!(#field_names)), #root::module::Parameter::as_nested_value(&self.#field_names));)*
                    parameters
                }

                fn parameters_mut(&mut self) -> #root::module::ModuleParamMut<'_> {
                    let mut parameters = #root::nested::NestedHashMap::new();
                    #(parameters.insert(std::rc::Rc::from(stringify!(#field_names)), #root::module::Parameter::as_nested_value_mut(&mut self.#field_names));)*
                    parameters
                }

                fn trainable_parameters(&self) -> #root::module::ModuleParamRef<'_> {
                    let mut parameters = #root::nested::NestedHashMap::new();
                    #(
                        if let Some(field) = #root::module::Parameter::as_trainable_nested_value(&self.#field_names) {
                            parameters.insert(std::rc::Rc::from(stringify!(#field_names)), field);
                        }
                    )*
                    parameters
                }

                fn all_frozen(&self) -> Option<bool> {
                    use #root::module::Parameter;
                    #(
                        if matches!(self.#field_names.is_frozen(), Some(false)) {
                            return Some(false);
                        }
                    )*
                    #default_all_frozen
                }

                fn any_frozen(&self) -> Option<bool> {
                    use #root::module::Parameter;
                    #(
                        if matches!(self.#field_names.is_frozen(), Some(true)) {
                            return Some(true);
                        }
                    )*
                    #default_any_frozen
                }
            }
        };
    }
}
