use crate::builder::BuilderStructAnalyzer;
use crate::shared::{attribute::AttributeItem, field::FieldTypeAnalyzer};
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Field, Ident};

pub struct BuilderAnalyzerFactory {}

pub trait BuilderAnalyzer {
    fn gen_new_fn(&self) -> TokenStream {
        quote! {}
    }
    fn gen_builder_fns(&self) -> TokenStream {
        quote! {}
    }
    fn gen_clone_impl(&self) -> TokenStream;
}

impl BuilderAnalyzerFactory {
    pub fn new() -> Self {
        Self {}
    }

    pub fn create_analyzer(&self, item: &syn::DeriveInput) -> Box<dyn BuilderAnalyzer> {
        let name = item.ident.clone();
        let builder_type = parse_asm(item);

        match builder_type {
            BuilderType::Struct(data) => Box::new(self.create_struct_analyzer(name, data)),
        }
    }

    fn create_struct_analyzer(&self, name: Ident, fields: Vec<Field>) -> BuilderStructAnalyzer {
        let fields = fields.into_iter().map(FieldTypeAnalyzer::new);

        let mut fields_required = Vec::new();
        let mut fields_option = Vec::new();
        let mut fields_default = Vec::new();

        for field in fields {
            let attributes: Vec<AttributeItem> = field
                .attributes()
                .filter(|attr| attr.has_name("builder"))
                .map(|attr| attr.item())
                .collect();

            if !attributes.is_empty() {
                let item = attributes.first().unwrap().clone();
                fields_default.push((field.clone(), item));
                continue;
            }

            if field.is_of_type(&["Option"]) {
                fields_option.push(field.clone());
                continue;
            }

            fields_required.push(field.clone());
        }

        BuilderStructAnalyzer::new(name, fields_required, fields_option, fields_default)
    }
}

enum BuilderType {
    Struct(Vec<Field>),
}

fn parse_asm(ast: &syn::DeriveInput) -> BuilderType {
    match &ast.data {
        syn::Data::Struct(struct_data) => {
            BuilderType::Struct(struct_data.fields.clone().into_iter().collect())
        }
        syn::Data::Enum(_) => panic!("Only struct can be derived"),
        syn::Data::Union(_) => panic!("Only struct can be derived"),
    }
}
