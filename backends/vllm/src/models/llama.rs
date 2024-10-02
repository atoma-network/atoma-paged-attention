use candle_core::{DType, Device, Tensor};
#[cfg(not(feature = "nccl"))]
use candle_nn::VarBuilder;
#[cfg(feature = "nccl")]
use cudarc::nccl::Comm;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use models::{
    llama::{Config, LlamaEosToks},
    FlashAttentionMetadata, Llama,
};
use std::path::Path;
#[cfg(feature = "nccl")]
use std::rc::Rc;
#[cfg(not(feature = "nccl"))]
use std::time::Instant;
#[cfg(not(feature = "nccl"))]
use tracing::info;

use crate::{
    model_executor::{
        Config as ModelConfig, ModelExecutor, ModelExecutorError, ModelFilePaths, ModelLoader,
        ModelLoaderError,
    },
    models::hub_load_safetensors,
};

impl ModelConfig for Config {
    fn alibi_slopes(&self) -> Option<&Tensor> {
        None
    }
    fn eos_token_ids(&self) -> Option<Vec<u32>> {
        match self.eos_token_id.clone() {
            None => None,
            Some(LlamaEosToks::Single(u)) => Some(vec![u]),
            Some(LlamaEosToks::Multiple(us)) => Some(us),
        }
    }
    fn hidden_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
    }
    fn sliding_window(&self) -> Option<usize> {
        None
    }
    fn softmax_scale(&self) -> f32 {
        1f32 / (self.hidden_dim() as f32).sqrt()
    }
}

/// Represents a Llama language model.
///
/// This struct encapsulates the configuration, device, data type, and the actual Llama model.
pub struct LlamaModel {
    /// The configuration for the Llama model.
    config: Config,
    /// The actual Llama model implementation.
    model: Llama,
}

impl ModelLoader for LlamaModel {
    type C = Config;

    fn fetch<T: AsRef<Path>>(
        api_key: String,
        cache_dir: T,
        model_id: String,
        revision: String,
    ) -> Result<ModelFilePaths, ModelLoaderError> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir.as_ref().to_path_buf())
            .build()?;

        let repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            revision,
        ));
        let config_file_path = repo.get("config.json")?;
        let tokenizer_file_path = repo.get("tokenizer.json")?;

        let model_weights_file_paths = if &model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0" {
            vec![repo.get("model.safetensors")?]
        } else {
            hub_load_safetensors(&repo, "model.safetensors.index.json")?
        };

        Ok(ModelFilePaths {
            config_path: config_file_path,
            tokenizer_path: tokenizer_file_path,
            weights_path: model_weights_file_paths,
        })
    }

    #[cfg(not(feature = "nccl"))]
    fn load(
        config: Self::C,
        device: &Device,
        dtype: DType,
        file_paths: &ModelFilePaths,
    ) -> Result<Self, ModelLoaderError>
    where
        Self: Sized,
    {
        info!("Loading Llama model ...");
        let start = Instant::now();

        let model = {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    file_paths.weights_path.as_slice(),
                    dtype,
                    device,
                )?
            };
            Llama::load(vb, &config, dtype, device)?
        };
        info!("Loaded Llama model in {:?}", start.elapsed());

        Ok(Self { model, config })
    }

    #[cfg(feature = "nccl")]
    fn load(
        _: Self::C,
        _: &Device,
        _: DType,
        _: &ModelFilePaths,
        _: &Rc<Comm>,
    ) -> Result<Self, ModelLoaderError>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}

impl ModelExecutor for LlamaModel {
    fn forward(
        &mut self,
        input: &Tensor,
        input_positions: &Tensor,
        selected_token_positions: &Tensor,
        kv_cache: Vec<&mut Tensor>,
        attention_metadata: FlashAttentionMetadata,
    ) -> Result<Tensor, ModelExecutorError> {
        Ok(self.model.forward(
            input,
            input_positions,
            selected_token_positions,
            &kv_cache,
            attention_metadata,
        )?)
    }

    fn config(&self) -> &Self::C {
        &self.config
    }
}
