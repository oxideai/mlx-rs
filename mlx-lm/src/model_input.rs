use tokenizers::Encoding;

pub struct ModelInputBuilder<C> {
    pub encodings: Vec<Encoding>,
    pub cache: C,
}

impl<C> ModelInputBuilder<C> {

}

pub trait ModelInput {
    fn from_builder<C>(builder: ModelInputBuilder<C>) -> Self;
}