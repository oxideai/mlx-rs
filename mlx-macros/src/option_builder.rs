use quote::quote;
use syn::ItemStruct;

pub(crate) fn expand_option_builder(
    input: &ItemStruct,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let struct_ident = &input.ident;
    let generics = &input.generics;
    let (field_idents, field_types): (Vec<_>, Vec<_>) = input
        .fields
        .iter()
        .map(|field| {
            let field_ident = field
                .ident
                .as_ref()
                .ok_or_else(|| syn::Error::new_spanned(field, "Field must have an identifier"))?;
            let field_type = field.ty.clone();
            Result::<_, syn::Error>::Ok((field_ident.clone(), field_type))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let fn_new_doc = format!("Create a new [`{}`] with the default values", struct_ident);
    let fn_builder_doc = format!(
        "Create a new [`{}`] builder with the default values",
        struct_ident
    );
    let fn_field_docs = field_idents
        .iter()
        .map(|field_ident| format!("Set the value of `{}`", field_ident));

    // Generate builder pattern implementation
    let output = quote! {
        #input

        impl #impl_generics #struct_ident #ty_generics #where_clause {
            #[doc = #fn_builder_doc]
            pub fn builder() -> crate::utils::Builder<Self> {
                crate::utils::Builder::new(Self::default())
            }

            #[doc = #fn_new_doc]
            pub fn new() -> Self {
                Self::default()
            }
        }

        impl #impl_generics crate::utils::Builder<#struct_ident #ty_generics> #where_clause {
            #(
                #[doc = #fn_field_docs]
                pub fn #field_idents(mut self, value: impl Into<#field_types>) -> Self {
                    self.inner.#field_idents = value.into();
                    self
                }
            )*
        }

        impl #impl_generics From<crate::utils::Builder<#struct_ident #ty_generics>> for #struct_ident #ty_generics #where_clause {
            fn from(builder: crate::utils::Builder<#struct_ident #ty_generics>) -> Self {
                builder.inner
            }
        }
    };

    Ok(output)
}