use std::io::Read;

use hf_hub::{api::sync::{Api, ApiBuilder, ApiRepo}, Repo};
use mlx_rs::{module::ModuleParameters, Array};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

mod model;

use model::{Mistral, ModelArgs};

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
    let model_args_filename = repo.get("params.json")?;
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;

    Ok(model_args)
}

fn load_weights(repo: &ApiRepo, model: &mut Mistral) -> Result<()> {
    let weights_filename = repo.get("weights.safetensors")?;
    let mut file = std::fs::File::open(weights_filename)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let safetensors = SafeTensors::deserialize(&buf)?;

    load_weights_from_safetensors(model, safetensors)
}

fn load_weights_from_safetensors(model: &mut Mistral, weights: SafeTensors<'_>) -> Result<()> {
    let params = model.parameters_mut().flatten();
    for (key, value) in params {
        let tensor = weights.tensor(&*key)?;
        *value = Array::try_from(tensor)?;
    }

    Ok(())
}

fn load_model(repo: &ApiRepo) -> Result<Mistral> {
    let model_args = get_model_args(repo)?;
    let mut model = Mistral::new(&model_args)?;
    load_weights(repo, &mut model)?;
    Ok(model)
}

fn main() -> Result<()> {
    // Put your huggingface access token in a .env file located at the root of
    // this example (ie. examples/mistral/.env)
    dotenv::dotenv().ok();
    let api = build_hf_api()?;

    // The model used in the original example is converted to safetensors and
    // uploaded to the huggingface hub
    let model_id = "minghuaw/Mistral-7B-v0.1".to_string();
    let repo = api.repo(Repo::new(model_id, hf_hub::RepoType::Model));
    let tokenizer = get_tokenizer(&repo)?;
    let mut model = load_model(&repo)?;

    Ok(())
}