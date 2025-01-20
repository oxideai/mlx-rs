use hf_hub::{api::sync::Api, Repo};
use tokenizers::Tokenizer;

mod model;

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T, E = Error> = std::result::Result<T, E>;

fn main() -> Result<()> {
    println!("Hello, world!");
    let api = Api::new()?;

    let model_id = "mistralai/Mistral-7B-v0.1".to_string();
    let repo = api.repo(Repo::new(model_id, hf_hub::RepoType::Model));
    let tokenizer_filename = repo.get("tokenizer.json")?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;
    let vocab_size = tokenizer.get_vocab_size(false);
    println!("Vocab size: {}", vocab_size);

    Ok(())
}
