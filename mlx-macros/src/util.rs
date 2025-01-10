pub(crate) struct FilteredFields<'a> {
    pub filtered: Vec<&'a syn::Field>,
    pub other_fields: Vec<&'a syn::Field>,
}

pub(crate) fn filter_fields_with_attr<'a>(
    fields: &'a syn::Fields,
    attr_name: &str,
) -> Result<FilteredFields<'a>, syn::Error> {
    let mut filtered = Vec::new();
    let mut other_fields = Vec::new();

    match fields {
        syn::Fields::Named(fields) => {
            for field in &fields.named {
                if field
                    .attrs
                    .iter()
                    .any(|attr| attr.path().is_ident(attr_name))
                {
                    filtered.push(field);
                } else {
                    other_fields.push(field);
                }
            }
        }
        syn::Fields::Unit => {}
        syn::Fields::Unnamed(_) => {
            return Err(syn::Error::new_spanned(
                fields,
                "Struct with unnamed fields is not supported".to_string(),
            ))
        }
    }

    Ok(FilteredFields {
        filtered,
        other_fields,
    })
}
