use std::path::Path;

use mlx_lm::{cache::ConcatKeyValueCache, models::gemma::gemma3::load_gemma3_model};
use mlx_lm_utils::tokenizer::{
    load_gemma_chat_template_from_file, ApplyChatTemplateArgs, Conversation, Role, Tokenizer,
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

const CACHED_TEST_MODEL_DIR: &str = "./cache/gemma-3-270m-bf16";

fn gemma3() -> anyhow::Result<()> {
    let model_dir = Path::new(CACHED_TEST_MODEL_DIR);

    let model_id = "mlx-community/gemma-3-270m-bf16".to_string();
    let tokenizer_file = model_dir.join("tokenizer.json");
    let chat_template_jinja_file = model_dir.join("chat_template.jinja");
    let mut tokenizer =
        Tokenizer::from_file(tokenizer_file).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let model_chat_template = load_gemma_chat_template_from_file(chat_template_jinja_file)?;

    let conversations = vec![Conversation {
        role: Role::User,
        content: "what's your name?",
    }];
    println!("Conversations: {:?}", conversations);

    let args = ApplyChatTemplateArgs {
        conversations: vec![conversations.into()],
        documents: None,
        model_id: &model_id,
        chat_template_id: None,
        add_generation_prompt: Some(true),
        continue_final_message: None,
        add_special_tokens: Some(true),
    };
    let encodings = tokenizer.apply_chat_template_and_encode(model_chat_template, args)?;
    let prompt: Vec<u32> = encodings
        .iter()
        .flat_map(|encoding| encoding.get_ids())
        .copied()
        .collect();
    println!("Prompt tokens (raw): {:?}", prompt);
    let prompt_tokens = Array::from(&prompt[..]).index(NewAxis);
    println!("Prompt tokens (array): {:?}", prompt_tokens);

    let mut cache = Vec::new();
    let mut model = load_gemma3_model(model_dir)?;
    let generate = mlx_lm::models::gemma::gemma3::Generate::<ConcatKeyValueCache>::new(
        &mut model,
        &mut cache,
        0.0,
        &prompt_tokens,
    );

    let mut tokens = Vec::new();
    for (token, ntoks) in generate.zip(0..256) {
        let token = token.unwrap();
        tokens.push(token.clone());

        if ntoks == 0 {
            eval(&tokens).unwrap();
        }

        if tokens.len() % 20 == 0 {
            eval(&tokens).unwrap();
            let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
            let s = tokenizer.decode(&slice, true).unwrap();
            print!("{s}");
        }
    }

    eval(&tokens).unwrap();
    let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
    let s = tokenizer.decode(&slice, true).unwrap();
    println!("{s}");

    println!("------");

    Ok(())
}

fn main() -> anyhow::Result<()> {
    gemma3()
}
