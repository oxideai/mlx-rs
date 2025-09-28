use hf_hub::{api::sync::ApiBuilder, Repo};
use mlx_lm::{cache::ConcatKeyValueCache, generate::Generate, models::qwen3::{self, load_qwen3_model, load_qwen3_tokenizer}};
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Conversation, Role, Tokenizer,
};
use mlx_rs::{array, ops::indexing::{IndexOp, NewAxis}, Array};

fn qwen3() -> anyhow::Result<()> {
    let api = ApiBuilder::new()
        .with_endpoint("https://hf-mirror.com".to_string())
        .with_cache_dir("../hf_cache".into())
        .build()?;

    let model_id = "mlx-community/Qwen3-4B-bf16".to_string();
    let repo = api.repo(Repo::new(model_id.clone(), hf_hub::RepoType::Model));
    let tokenizer_file = repo.get("tokenizer.json")?;
    let tokenizer_config_file = repo.get("tokenizer_config.json")?;
    let mut tokenizer =
        Tokenizer::from_file(tokenizer_file).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let model_chat_template = load_model_chat_template_from_file(tokenizer_config_file)?
        .expect("Model chat template not found");

    let conversations = vec![Conversation {
        role: Role::User,
        content: "hello",
    }];
    let args = ApplyChatTemplateArgs {
        conversations: vec![conversations.into()],
        documents: None,
        model_id: &model_id,
        chat_template_id: None,
        add_generation_prompt: None,
        continue_final_message: None,
    };
    let encodings = tokenizer.apply_chat_template_and_encode(model_chat_template, args)?;
    let prompt: Vec<u32> = encodings
        .iter()
        .map(|encoding| encoding.get_ids())
        .flatten()
        .copied()
        .collect();
    let tokens = Array::from(&prompt[..]).index(NewAxis);

    let model = load_qwen3_model(&repo)?;

    let generate = Generate::builder()
        .model::<_, qwen3::ModelInput<ConcatKeyValueCache>>(model)
        .prompt(tokens)
        .tokenizer(tokenizer)
        .build();

    Ok(())
}

fn main() -> anyhow::Result<()> {
    qwen3()
}
