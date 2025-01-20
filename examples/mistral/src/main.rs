use hf_hub::{api::sync::{Api, ApiBuilder}, Repo};
use tokenizers::Tokenizer;

mod model;

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T, E = Error> = std::result::Result<T, E>;

fn main() -> Result<()> {
    // Put your huggingface access token in a .env file located at the root of
    // this example (ie. examples/mistral/.env)
    dotenv::dotenv().ok();
    let hf_token = std::env::var("HF_TOKEN")?;

    let api = ApiBuilder::new()
        .with_token(Some(hf_token))
        .with_cache_dir("./data".into())
        .build()?;

    let model_id = "mistralai/Mistral-7B-v0.1".to_string();
    let repo = api.repo(Repo::new(model_id, hf_hub::RepoType::Model));
    let tokenizer_filename = repo.get("tokenizer.json")?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;
    let vocab_size = tokenizer.get_vocab_size(false);

    Ok(())
}
