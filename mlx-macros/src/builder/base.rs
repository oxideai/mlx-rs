use super::BuilderAnalyzerFactory;
use quote::quote;

pub(crate) fn derive_impl(item: &syn::DeriveInput) -> proc_macro::TokenStream {
    let factory = BuilderAnalyzerFactory::new();
    let analyzer = factory.create_analyzer(item);

    let constructor = analyzer.gen_new_fn();
    let builders = analyzer.gen_builder_fns();
    let clone = analyzer.gen_clone_impl();

    quote! {
        #constructor
        #builders
        #clone
    }
    .into()
}
