use quote::ToTokens;

pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) struct MandatoryField {
    pub ident: syn::Ident,
    pub ty: syn::Type,
}

pub(crate) struct OptionalField {
    pub ident: syn::Ident,
    pub ty: syn::Type,
    pub default: syn::Path,
}

#[derive(Debug, Clone)]
pub(crate) enum PathOrIdent {
    Path(syn::Path),
    Ident(syn::Ident),
}

impl ToTokens for PathOrIdent {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            PathOrIdent::Path(path) => path.to_tokens(tokens),
            PathOrIdent::Ident(ident) => ident.to_tokens(tokens),
        }
    }
}