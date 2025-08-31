#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    RenderTemplate(#[from] minijinja::Error),
}