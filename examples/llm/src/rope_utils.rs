use std::{borrow::Borrow, collections::HashMap};

use mlx_rs::{builder::Builder, error::Exception, macros::ModuleParameters, nn};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq)]
pub enum FloatOrStr<'a> {
    Float(f32),
    Str(&'a str),
}

// TODO: check if additionl serde attributes are needed
#[derive(Debug, Clone, Deserialize)]
pub enum FloatOrString {
    Float(f32),
    String(String),
}

impl FloatOrString {
    pub fn borrowed(&self) -> FloatOrStr {
        match self {
            FloatOrString::Float(f) => FloatOrStr::Float(*f),
            FloatOrString::String(s) => FloatOrStr::Str(s),
        }
    }
}

pub fn initialize_rope(
    dims: i32,
    base: f32, // rope_theta
    traditional: bool,
    scaling_config: &Option<HashMap<String, FloatOrString>>,
    max_position_embeddings: i32,
) -> Result<nn::Rope, Exception> {
    let rope_type = scaling_config
        .as_ref()
        .and_then(|config| {
            config.get("type")
                .or_else(|| config.get("rope_type"))
                .map(FloatOrString::borrowed)
        })
        .unwrap_or(FloatOrStr::Str("default"));

    if rope_type == FloatOrStr::Str("default") ||
        rope_type == FloatOrStr::Str("linear") 
    {
        let scale = if rope_type == FloatOrStr::Str("linear") {
            let den = match scaling_config.as_ref().and_then(|config| config.get("factor"))
                .map(FloatOrString::borrowed)
                .ok_or_else(|| Exception::custom(r#"key "factor" is not found in scaling config"#))? {
                FloatOrStr::Float(f) => f,
                FloatOrStr::Str(s) => s.parse::<f32>().map_err(|_| {
                    Exception::custom(r#"key "factor" is not a valid float"#)
                })?,
            };

            1.0 / den
        } else {
            1.0
        };

        let rope = nn::RopeBuilder::new(dims)
            .traditional(traditional)
            .base(base)
            .scale(scale)
            .build()
            .expect("Infallible");
        return Ok(rope)
    } else if rope_type == FloatOrStr::Str("llama3") {
        todo!()
    } else if rope_type == FloatOrStr::Str("yarn") {
        todo!()
    } else if rope_type == FloatOrStr::Str("longrope") {
        todo!()
    }

    Err(Exception::custom(format!("Unsupported RoPE type {:?}", rope_type)))
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Llama3Rope {
    dims: i32,
    max_position_embeddings: i32,
    traditional: bool,
}