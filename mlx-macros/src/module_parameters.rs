use syn::{DataStruct, DeriveInput, Generics, Ident};

pub(crate) fn expand_module_parameters(
    input: &DeriveInput,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let struct_ident = &input.ident;
    let generics = &input.generics;
    match &input.data {
        syn::Data::Struct(data) => {
            expand_module_parameters_for_struct(struct_ident, generics, data)
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

    Ok(impl_module_parameters_for_struct(ident, generics, fields))
}

fn impl_module_parameters_for_struct(
    ident: &Ident,
    generics: &Generics,
    fields: Vec<&syn::Field>,
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
        impl #impl_generics _mlx_rs::module::ModuleParameters for #ident #ty_generics #where_clause {
            fn freeze_parameters(&mut self, recursive: bool) {
                use _mlx_rs::module::Parameter;
                #(self.#field_names.freeze(recursive);)*
            }

            fn unfreeze_parameters(&mut self, recursive: bool) {
                use _mlx_rs::module::Parameter;
                #(self.#field_names.unfreeze(recursive);)*
            }

            fn parameters(&self) -> _mlx_rs::module::ModuleParamRef<'_> {
                let mut parameters = _mlx_rs::nested::NestedHashMap::new();
                #(parameters.insert(stringify!(#field_names), _mlx_rs::module::Parameter::as_nested_value(&self.#field_names));)*
                parameters
            }

            fn parameters_mut(&mut self) -> _mlx_rs::module::ModuleParamMut<'_> {
                let mut parameters = _mlx_rs::nested::NestedHashMap::new();
                #(parameters.insert(stringify!(#field_names), _mlx_rs::module::Parameter::as_nested_value_mut(&mut self.#field_names));)*
                parameters
            }

            fn trainable_parameters(&self) -> _mlx_rs::module::ModuleParamRef<'_> {
                let mut parameters = _mlx_rs::nested::NestedHashMap::new();
                #(
                    if let Some(field) = _mlx_rs::module::Parameter::as_trainable_nested_value(&self.#field_names) {
                        parameters.insert(stringify!(#field_names), field);
                    }
                )*
                parameters
            }

            fn all_frozen(&self) -> Option<bool> {
                use _mlx_rs::module::Parameter;
                #(
                    if matches!(self.#field_names.is_frozen(), Some(false)) {
                        return Some(false);
                    }
                )*
                #default_all_frozen
            }

            fn any_frozen(&self) -> Option<bool> {
                use _mlx_rs::module::Parameter;
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
