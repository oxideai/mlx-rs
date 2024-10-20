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

    let mut optional_field_idents = Vec::new();
    let mut optional_field_types = Vec::new();
    let mut optional_field_defaults = Vec::new();
    let mut mandatory_field_idents = Vec::new();
    let mut mandatory_field_types = Vec::new();

    for field in input.fields.iter() {
        if attrs_contains_optional(&field.attrs) {
            let field_attr = FieldAttr::from_attributes(&field.attrs)?;
            optional_field_idents.push(
                field
                    .ident
                    .as_ref()
                    .ok_or("Only named fields are supported")?,
            );
            optional_field_types.push(&field.ty);
            if generate_build_fn {
                optional_field_defaults.push(
                    field_attr
                        .default_value
                        .ok_or("Default value is required for optional fields")?,
                );
            }
        } else {
            mandatory_field_idents.push(&field.ident);
            mandatory_field_types.push(&field.ty);
        }
    }

    let builder_ident = Ident::new(&format!("{}Builder", struct_ident), struct_ident.span());

    let builder_struct = quote! {
        #[derive(Debug, Clone, Default)]
        pub struct #builder_ident {
            #(
                pub #optional_field_idents: Option<#optional_field_types>,
            )*
        }
    };

    let builder_new_doc = format!("Create a new [`{}`]", builder_ident);
    let struct_builder_doc = format!(
        "Create a new [`{}`] builder with the default values",
        struct_ident
    );

    let builder_init = quote! {
        impl #builder_ident {
            #[doc = #builder_new_doc]
            pub fn new() -> Self {
                Self::default()
            }
        }

        impl #struct_ident {
            #[doc = #struct_builder_doc]
            pub fn builder() -> #builder_ident {
                #builder_ident::new()
            }
        }
    };

    let builder_setter_docs = optional_field_idents
        .iter()
        .map(|field_ident| format!("Set the value of `{:?}`", field_ident));

    let builder_setters = quote! {
        impl #builder_ident {
            #(
                #[doc = #builder_setter_docs]
                pub fn #optional_field_idents(mut self, value: impl Into<#optional_field_types>) -> Self {
                    self.#optional_field_idents = Some(value.into());
                    self
                }
            )*
        }
    };

    let builder_build_doc = format!("Build a new [`{}`]", struct_ident);
    let builder_build = if generate_build_fn {
        quote! {
            impl #builder_ident {
                #[doc = #builder_build_doc]
                pub fn build(self, #(#mandatory_field_idents: #mandatory_field_types),*) -> #struct_ident {
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
        }
    } else {
        quote! {}
    };

    Ok(quote! {
        #builder_struct
        #builder_init
        #builder_setters
        #builder_build
    })
}
