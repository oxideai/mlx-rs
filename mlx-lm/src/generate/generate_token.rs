use std::marker::PhantomData;

use mlx_rs::{error::Exception, module::Module, Array};

use crate::{
    cache::KeyValueCache, sampler::Sampler, utils::try_unwrap, ModelInput, ModelInputBuilder, ModelOutput
};

pub(super) enum Stage<C, T> {
    Generating,
    Prefill {
        prompt: Array,
        state: T,
    },
    Decode {
        y: Array,
        cache: Vec<Option<C>>,
        state: T,
    },
}

impl<C, T> Stage<C, T> {
    fn take(&mut self) -> Self {
        debug_assert!(!matches!(self, Self::Generating));

        let mut swap = Self::Generating;
        std::mem::swap(self, &mut swap);
        swap
    }
}

pub(super) struct GenerateToken<M, I, S, C, T> {
    pub model: M,
    pub model_input_marker: PhantomData<I>,
    pub sampler: S,
    pub temp: f32,
    pub stage: Stage<C, T>,
}

impl<M, I, S, C, T> Iterator for GenerateToken<M, I, S, C, T>
where
    M: Module<I>,
    M::Error: Into<Exception>,
    M::Output: ModelOutput,
    for<'input> I: ModelInput<'input, C, T>,
    S: Sampler,
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            model,
            model_input_marker: _,
            temp,
            sampler,
            stage,
        } = self;

        match stage.take() {
            Stage::Prefill { prompt, mut state } => {
                let mut cache = Vec::new();
                let builder = ModelInputBuilder {
                    y: &prompt,
                    cache: &mut cache,
                    state: &mut state,
                };
                let input = I::from_model_input_builder(builder);
                let output = try_unwrap!(model.forward(input));
                let logits = output.logits();
                let y = try_unwrap!(sampler.sample(logits, *temp));

                *stage = Stage::Decode {
                    y: y.clone(),
                    cache,
                    state,
                };

                Some(Ok(y))
            }
            Stage::Decode { y, mut cache, mut state } => {
                let builder = ModelInputBuilder {
                    y: &y,
                    cache: &mut cache,
                    state: &mut state,
                };

                let input = I::from_model_input_builder(builder);
                let output = try_unwrap!(model.forward(input));
                let logits = output.logits();
                let y = try_unwrap!(sampler.sample(logits, *temp));

                *stage = Stage::Decode {
                    y: y.clone(),
                    cache,
                    state,
                };

                Some(Ok(y))
            },
            Stage::Generating => unreachable!(),
        }
    }
}
