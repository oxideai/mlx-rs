use darling::FromDeriveInput;
use syn::{DataStruct, DeriveInput, Generics, Ident};

#[derive(Debug, Clone, FromDeriveInput)]
#[darling(attributes(module))]
#[allow(dead_code)]
pub(crate) struct StructProperty {
    /// Rename `mlx_rs` if Some(_)
    pub root: Option<syn::Path>,
}

pub(crate) fn expand_module_parameters(
    input: &DeriveInput,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let prop = StructProperty::from_derive_input(input)?;
    let root = match prop.root {
        Some(path) => path,
        None => syn::parse_quote!(::mlx_rs),
    };
    let struct_ident = &input.ident;
    let generics = &input.generics;
    match &input.data {
        syn::Data::Struct(data) => {
            expand_module_parameters_for_struct(struct_ident, generics, data, root)
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
    root: syn::Path,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let fields = match &data.fields {
        syn::Fields::Named(fields) => {
            // filter out fields with #[param]
            fields
                .named
                .iter()
                .filter(|field| field.attrs.iter().any(|attr| attr.path().is_ident("param")))
                .collect()
        }
        syn::Fields::Unit => vec![],
        syn::Fields::Unnamed(_) => {
            return Err(syn::Error::new_spanned(
                ident,
                "ModuleParameters cannot be derived for structs with unnamed fields",
            ))
        }
    };

    Ok(impl_module_parameters_for_struct(
        ident, generics, fields, root,
    ))
}

fn impl_module_parameters_for_struct(
    ident: &Ident,
    generics: &Generics,
    fields: Vec<&syn::Field>,
    root: syn::Path,
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

    quote::quote! {
        impl #impl_generics #root::module::ModuleParameters for #ident #ty_generics #where_clause {
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
    }
}
