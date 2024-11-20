use super::BuilderAnalyzer;
use crate::shared::{attribute::AttributeItem, field::FieldTypeAnalyzer};
use proc_macro2::{Ident, TokenStream};
use quote::quote;

pub struct BuilderStructAnalyzer {
    name: Ident,
    fields_required: Vec<FieldTypeAnalyzer>,
    fields_option: Vec<FieldTypeAnalyzer>,
    fields_default: Vec<(FieldTypeAnalyzer, AttributeItem)>,
}

impl BuilderStructAnalyzer {
    pub fn new(
        name: Ident,
        fields_required: Vec<FieldTypeAnalyzer>,
        fields_option: Vec<FieldTypeAnalyzer>,
        fields_default: Vec<(FieldTypeAnalyzer, AttributeItem)>,
    ) -> Self {
        Self {
            name,
            fields_required,
            fields_option,
            fields_default,
        }
    }

    fn wrap_impl_block(&self, tokens: TokenStream) -> TokenStream {
        let name = &self.name;

        quote! {
            impl #name {
                #tokens
            }
        }
    }

    fn names(&self) -> Vec<FieldTypeAnalyzer> {
        let mut names = Vec::new();

        for field in self.fields_required.iter() {
            names.push(field.clone());
        }

        for field in self.fields_option.iter() {
            names.push(field.clone());
        }

        for (field, _) in self.fields_default.iter() {
            names.push(field.clone());
        }

        names
    }
}

impl BuilderAnalyzer for BuilderStructAnalyzer {
    fn gen_new_fn(&self) -> TokenStream {
        let mut body = quote! {};
        let mut names = Vec::new();

        for field in self.fields_required.iter() {
            let name = field.ident();
            let ty = &field.field.ty;

            body.extend(quote! {
                #name: #name,
            });
            names.push(quote! {
                #name: #ty
            });
        }

        for field in self.fields_option.iter() {
            let name = field.ident();

            body.extend(quote! {
                #name: None,
            });
        }

        for (field, attribute) in self.fields_default.iter() {
            let name = field.ident();
            let value = &attribute.value;
            match value {
                syn::Lit::Str(value) => {
                    let stream: proc_macro2::TokenStream = value.value().parse().unwrap();

                    body.extend(quote! {
                        #name: #stream,
                    });
                }
                _ => {
                    body.extend(quote! {
                        #name: #value,
                    });
                }
            };
        }

        let body = quote! {
            /// Create a new instance of the builder.
            pub fn new(
                #(#names),*
            ) -> Self {
                Self { #body }
            }
        };
        self.wrap_impl_block(body)
    }

    fn gen_builder_fns(&self) -> TokenStream {
        let mut body = quote! {};

        for (field, _) in self.fields_default.iter() {
            let name = field.ident();
            let doc = field.doc().unwrap_or_else(|| {
                quote! {
                        /// Set the default value for the field.
                }
            });
            let ty = &field.field.ty;
            let fn_name = Ident::new(&format!("with_{name}"), name.span());

            body.extend(quote! {
                #doc
                pub fn #fn_name(mut self, #name: #ty) -> Self {
                    self.#name = #name;
                    self
                }
            });
        }

        for field in self.fields_option.iter() {
            let name = field.ident();
            let ty = &field.field.ty;
            let fn_name = Ident::new(&format!("with_{name}"), name.span());

            body.extend(quote! {
                /// Set the default value for the field.
                pub fn #fn_name(mut self, #name: #ty) -> Self {
                    self.#name = #name;
                    self
                }
            });
        }

        self.wrap_impl_block(body)
    }

    fn gen_clone_impl(&self) -> TokenStream {
        let name = &self.name;
        let names = self.names().into_iter().map(|name| {
            let name = name.ident();
            quote! { #name: self.#name.clone() }
        });

        quote! {
            impl Clone for #name {
                fn clone(&self) -> Self {
                    Self {
                        #(#names),*
                    }
                }
            }

        }
    }
}
