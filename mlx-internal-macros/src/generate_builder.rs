use darling::FromAttributes;
use quote::quote;
use syn::{Attribute, Ident, ItemStruct, Path};

#[derive(Debug, FromAttributes)]
#[darling(attributes(generate_builder))]
struct StructAttr {
    generate_build_fn: Option<bool>,
}

#[derive(Debug, FromAttributes)]
#[darling(attributes(optional))]
struct FieldAttr {
    default_value: Option<Path>,
    skip: Option<bool>,
}

fn attrs_contains_optional(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| attr.path().is_ident("optional"))
}

pub(crate) fn expand_generate_builder(
    input: &ItemStruct,
) -> Result<proc_macro2::TokenStream, Box<dyn std::error::Error>> {
    let generate_build_fn = StructAttr::from_attributes(&input.attrs)?
        .generate_build_fn
        .unwrap_or(true);
    let struct_ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let mut optional_field_idents = Vec::new();
    let mut optional_field_types = Vec::new();
    let mut optional_field_defaults = Vec::new();
    let mut optional_field_skip = Vec::new();
    let mut mandatory_field_idents = Vec::new();
    let mut mandatory_field_types = Vec::new();

    for field in input.fields.iter() {
        if attrs_contains_optional(&field.attrs) {
            let field_attr = FieldAttr::from_attributes(&field.attrs)?;
            let skip = field_attr.skip.unwrap_or(false);
            if skip && generate_build_fn {
                return Err("Skip is not allowed when build function is generated".into());
            }
            optional_field_skip.push(skip);

            optional_field_idents.push(
                field
                    .ident
                    .as_ref()
                    .ok_or("Only named fields are supported")?,
            );
            optional_field_types.push(&field.ty);
            if generate_build_fn {
                optional_field_defaults.push(field_attr.default_value);
            }
        } else {
            mandatory_field_idents.push(&field.ident);
            mandatory_field_types.push(&field.ty);
        }
    }

    let builder_ident = Ident::new(&format!("{}Builder", struct_ident), struct_ident.span());
    let modified_optional_field_types = optional_field_types
        .iter()
        .zip(optional_field_skip.iter())
        .map(|(field_type, skip)| {
            if !skip {
                quote! { Option<#field_type> }
            } else {
                quote! { #field_type }
            }
        });

    let builder_struct_doc = format!("Builder for [`{}`]", struct_ident);
    let field_doc = format!("See [`{}`] for more details", struct_ident);
    let builder_struct = quote! {
        #[doc = #builder_struct_doc]
        #[derive(Debug, Clone, Default)]
        pub struct #builder_ident #ty_generics #where_clause {
            #(
                #[doc = #field_doc]
                pub #optional_field_idents: #modified_optional_field_types,
            )*
        }
    };

    let builder_new_doc = format!("Create a new [`{}`]", builder_ident);
    let struct_builder_doc = format!(
        "Create a new [`{}`] builder with the default values",
        struct_ident
    );

    let builder_init = quote! {
        impl #impl_generics #builder_ident #ty_generics #where_clause {
            #[doc = #builder_new_doc]
            pub fn new() -> Self {
                Self::default()
            }
        }

        impl #impl_generics #struct_ident #ty_generics #where_clause {
            #[doc = #struct_builder_doc]
            pub fn builder() -> #builder_ident #ty_generics {
                #builder_ident::new()
            }
        }
    };

    let builder_setter_docs = optional_field_idents
        .iter()
        .zip(optional_field_skip.iter())
        .filter_map(|(field_ident, skip)| {
            if !skip {
                Some(format!("Set the value of `{:?}`", field_ident))
            } else {
                None
            }
        });
    let filtered_optional_field_idents = optional_field_idents
        .iter()
        .zip(optional_field_skip.iter())
        .filter_map(|(field_ident, skip)| if !skip { Some(field_ident) } else { None });
    let filtered_optional_field_types = optional_field_types
        .iter()
        .zip(optional_field_skip.iter())
        .filter_map(|(field_type, skip)| if !skip { Some(field_type) } else { None });

    let builder_setters = quote! {
        impl #impl_generics #builder_ident #ty_generics #where_clause {
            #(
                #[doc = #builder_setter_docs]
                pub fn #filtered_optional_field_idents(mut self, value: impl Into<#filtered_optional_field_types>) -> Self {
                    self.#filtered_optional_field_idents = Some(value.into());
                    self
                }
            )*
        }
    };

    let builder_build = if generate_build_fn {
        let builder_build_doc = format!("Build a new [`{}`]", struct_ident);
        let struct_new_doc = format!("Create a new [`{}`] with default values", struct_ident);
        let optional_field_defaults: Vec<_> = optional_field_defaults
            .iter()
            .map(|default| {
                default
                    .clone()
                    .ok_or("Default value must be supplied to generate build function")
            })
            .collect::<Result<Vec<_>, _>>()?;

        quote! {
            impl #impl_generics #builder_ident #ty_generics #where_clause {
                #[doc = #builder_build_doc]
                pub fn build(self, #(#mandatory_field_idents: #mandatory_field_types),*) -> #struct_ident #ty_generics {
                    #struct_ident {
                        #(
                            #mandatory_field_idents,
                        )*
                        #(
                            #optional_field_idents: self.#optional_field_idents.unwrap_or_else(|| #optional_field_defaults),
                        )*
                    }
                }
            }

            impl #impl_generics #struct_ident #ty_generics #where_clause {
                #[doc = #struct_new_doc]
                pub fn new(#(#mandatory_field_idents: #mandatory_field_types),*) -> Self {
                    Self::builder().build(#(#mandatory_field_idents),*)
                }
            }
        }
    } else {
        quote! {}
    };

    // Only implement Default trait if no mandatory fields are present
    let default_impl = if mandatory_field_idents.is_empty() && generate_build_fn {
        quote! {
            impl #impl_generics Default for #struct_ident #ty_generics #where_clause {
                fn default() -> Self {
                    Self::new()
                }
            }
        }
    } else {
        quote! {}
    };

    Ok(quote! {
        #builder_struct
        #builder_init
        #builder_setters
        #builder_build
        #default_impl
    })
}
