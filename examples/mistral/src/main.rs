use hf_hub::{api::sync::{Api, ApiBuilder, ApiRepo}, Repo};
use tokenizers::Tokenizer;

mod config;
mod model_args;
mod model;

use config::Config;
use model_args::ModelArgs;

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T, E = Error> = std::result::Result<T, E>;

fn build_hf_api() -> Result<Api> {
    // Put your huggingface access token in a .env file located at the root of
    // this example (ie. examples/mistral/.env)
    dotenv::dotenv().ok();
    let hf_token = std::env::var("HF_TOKEN").ok();
    let cache_dir = std::env::var("HF_CACHE_DIR").ok();

    let mut builder = ApiBuilder::new()
        .with_token(hf_token);
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir.into());
    }
    builder.build().map_err(Into::into)
}

fn get_tokenizer(repo: &ApiRepo) -> Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;
    Tokenizer::from_file(tokenizer_filename)
}

fn get_model_args(repo: &ApiRepo) -> Result<ModelArgs> {
    let model_args_filename = repo.get("config.json")?;
    let file = std::fs::File::open(model_args_filename)?;

    // config.json file from HF is a bit different from the params.json file
    // from mistral.
    let config: Config = serde_json::from_reader(file)?;
    let model_args = ModelArgs::from(config);

    Ok(model_args)
}

fn main() -> Result<()> {
    // Put your huggingface access token in a .env file located at the root of
    // this example (ie. examples/mistral/.env)
    dotenv::dotenv().ok();
    let api = build_hf_api()?;

    let model_id = "mistralai/Mistral-7B-v0.1".to_string();
    let repo = api.repo(Repo::new(model_id, hf_hub::RepoType::Model));
    let tokenizer = get_tokenizer(&repo)?;

    let model_args = get_model_args(&repo)?;
    println!("{:?}", model_args);

    Ok(())
}
