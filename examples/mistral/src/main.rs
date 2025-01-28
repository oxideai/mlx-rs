use std::io::Read;

use hf_hub::{
    api::sync::{Api, ApiBuilder, ApiRepo},
    Repo,
};
use mlx_rs::{
    array,
    module::{Module, ModuleParameters},
    ops::indexing::argmax,
    prelude::{IndexOp, NewAxis},
    random::categorical,
    transforms::eval,
    Array,
};
use safetensors::SafeTensors;
use tokenizers::{processors::template::TemplateProcessing, AddedToken, Tokenizer};

mod model;

use model::{Mistral, MistralInput, MistralOutput, ModelArgs};

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T, E = Error> = std::result::Result<T, E>;

fn build_hf_api() -> Result<Api> {
    // Put your huggingface access token in a .env file located at the root of
    // this example (ie. examples/mistral/.env)
    dotenv::dotenv().ok();
    let hf_token = std::env::var("HF_TOKEN").ok();
    let cache_dir = std::env::var("HF_CACHE_DIR").ok();

    let mut builder = ApiBuilder::new().with_token(hf_token);
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir.into());
    }
    builder.build().map_err(Into::into)
}

fn get_tokenizer(repo: &ApiRepo) -> Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let mut t = Tokenizer::from_file(tokenizer_filename)?;
    t.add_special_tokens(&[AddedToken::from(String::from("<s>"), true)]);
    let post_processor = TemplateProcessing::builder()
        .try_single("[BOS] $A")?
        .special_tokens(vec![("[BOS]", 1)])
        .build()?;
    t.with_post_processor(Some(post_processor));
    Ok(t)
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

fn sample(logits: &Array, temp: Option<f32>) -> Result<Array> {
    match temp {
        None | Some(0.0) => argmax(logits, -1, None).map_err(Into::into),
        Some(temp) => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical(logits, None, None, None).map_err(Into::into)
        }
    }
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

struct Generate<'a> {
    model: &'a mut Mistral,
    temp: Option<f32>,
    state: GenerateState<'a>,
}

enum GenerateState<'a> {
    Start {
        prompt_token: &'a Array,
    },
    Continue {
        y: Array,
        cache: Vec<Option<(Array, Array)>>,
    },
}

impl<'a> Generate<'a> {
    pub fn new(model: &'a mut Mistral, prompt_token: &'a Array, temp: Option<f32>) -> Self {
        Self {
            model,
            temp,
            state: GenerateState::Start { prompt_token },
        }
    }
}

impl<'a> Iterator for Generate<'a> {
    type Item = Result<Array>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Start { prompt_token } => {
                let initial_cache = Vec::with_capacity(0); // This won't allocate
                let input = MistralInput {
                    inputs: &*prompt_token,
                    cache: &initial_cache,
                };
                let MistralOutput { logits, cache } = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));

                self.state = GenerateState::Continue {
                    y: y.clone(),
                    cache,
                };

                Some(Ok(y))
            }
            GenerateState::Continue { y, cache } => {
                let next_token = y.index((.., NewAxis));
                let input = MistralInput {
                    inputs: &next_token,
                    cache: cache.as_slice(),
                };
                let MistralOutput {
                    logits,
                    cache: new_cache,
                } = tri!(self.model.forward(input));

                let logits = tri!(logits.squeeze(&[1]));
                let y = tri!(sample(&logits, self.temp));

                self.state = GenerateState::Continue {
                    y: y.clone(),
                    cache: new_cache,
                };

                Some(Ok(y))
            }
        }
    }
}

fn main() -> Result<()> {
    // Put your huggingface access token in a .env file located at the root of
    // this example (ie. examples/mistral/.env)
    dotenv::dotenv().ok();
    let api = build_hf_api()?;

    // Parse args
    // TODO: take args from command line with clap
    let temp = None;
    let max_tokens = 20;
    let tokens_per_eval = 10;
    let seed = 13;

    mlx_rs::random::seed(seed)?;

    // The model used in the original example is converted to safetensors and
    // uploaded to the huggingface hub
    let model_id = "minghuaw/Mistral-7B-v0.1".to_string();
    let repo = api.repo(Repo::new(model_id, hf_hub::RepoType::Model));
    let tokenizer = get_tokenizer(&repo)?;
    let mut model = load_model(&repo)?;

    model = mlx_rs::nn::quantize(model, None, None)?;

    let prompt = "hello world";
    let encoding = tokenizer.encode(prompt, false)?;
    let encoding = tokenizer.post_process(encoding, None, true)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);

    let generate = Generate::new(&mut model, &prompt_tokens, temp);
    let mut tokens = Vec::with_capacity(max_tokens);
    for (token, ntoks) in generate.zip(0..max_tokens) {
        let token = token?;
        tokens.push(token);

        if ntoks == 0 {
            eval(&tokens)?;
            // TODO: timing
        }

        if tokens.len() & tokens_per_eval == 0 {
            eval(&tokens)?;
            let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
            let s = tokenizer.decode(&slice, true)?;
            print!("{}", s);
        }
    }

    eval(&tokens)?;
    let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
    let s = tokenizer.decode(&slice, true)?;
    println!("{}", s);

    println!("-------------------");

    Ok(())
}
