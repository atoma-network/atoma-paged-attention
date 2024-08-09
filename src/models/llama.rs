use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};
use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use serde::{Deserialize, Serialize};

use crate::flash_attention::{FlashAttention, FlashAttentionMetadata};

/// Maximum input sequence token length
const MAX_SEQ_LEN: usize = 4096;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl Config {
    pub fn config_7b_v1() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
        }
    }

    pub fn config_7b_v2() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
        }
    }
}

#[derive(Clone, Debug)]
/// Cache for Llama model
pub struct Cache {
    cos: Tensor,
    sin: Tensor,
}

impl Cache {
    /// Constructor
    pub fn new(config: &Config, device: &Device, dtype: DType) -> Result<Self> {
        // Precomputed frequency tensor for complex exponentials (cis)
        let n_elem = config.hidden_size / config.num_attention_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }
}

struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    span: tracing::Span,
    span_rot: tracing::Span,
    cos_sin_cache: Cache,
    attention: FlashAttention,
}

impl CausalSelfAttention {
    fn apply_rotary_embed(&self, x: &Tensor, input_positions: &Tensor) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (b_sz, _num_heads, num_total_tokens, _hidden_size) = x.dims4()?; // [1, num_heads, num_total_tokens, hidden_size]

        if b_sz != 1 {
            candle_core::bail!("batch size must be 1, got {}", b_sz);
        }
        if input_positions.dims() != [1, num_total_tokens] {
            candle_core::bail!(
            "index_positions must be of shape [batch_size, sequence_length] = [{}, {}], got {:?}",
            b_sz,
            num_total_tokens,
            input_positions.dims()
        );
        }
        if input_positions.dtype() != DType::I64 {
            candle_core::bail!(
                "input_positions must be of dtype i64, got {:?}",
                input_positions.dtype()
            );
        }

        // select input positions tokens
        let cos = self
            .cos_sin_cache
            .cos
            .index_select(&input_positions.flatten(0, 1)?, 0)?;
        let sin = self
            .cos_sin_cache
            .sin
            .index_select(&input_positions.flatten(0, 1)?, 0)?;

        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(
        &mut self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_cache: &Tensor,
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let (batch_size, num_total_tokens, _hidden_size) = x.dims3()?;
        if batch_size != 1 {
            candle_core::bail!(
                "x must be of shape [1, num_total_tokens], got {:?}",
                x.dims()
            );
        }

        let _enter = self.span.enter();
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((
                batch_size,
                num_total_tokens,
                self.num_attention_heads,
                self.head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((
                batch_size,
                num_total_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.reshape((
            batch_size,
            num_total_tokens,
            self.num_key_value_heads,
            self.head_dim,
        ))?;

        let q = self.apply_rotary_embed(&q, input_positions)?;
        let k = self.apply_rotary_embed(&k, input_positions)?;

        // transpose the matrices back to [sequence_length, num_heads, head_dim]
        let q = q.transpose(1, 2)?.squeeze(0)?.contiguous()?;
        let k = k.transpose(1, 2)?.squeeze(0)?.contiguous()?;
        let v = v.squeeze(0)?;

        let o = self
            .attention
            .forward(&q, &k, &v, kv_cache, attention_metadata)?;

        let o = o.unsqueeze(0)?;
        let out = self.o_proj.forward(&o)?;

        Ok(out)
    }

    fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: head_dim,
            span,
            span_rot,
            attention: FlashAttention::new(
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                head_dim,
                1f32 / (head_dim as f32).sqrt(),
                None,
                None,
                dtype,
                device.clone(),
            )?,
            cos_sin_cache: Cache::new(&cfg, device, dtype)?,
        })
    }
}

#[derive(Clone, Debug)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &mut self,
        x: &Tensor,
        input_positions: &Tensor,
        cache: &Tensor,
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(&x)?;
        let x = (self
            .attn
            .forward(&x, input_positions, cache, attention_metadata)?
            + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg, dtype, device)?;
        let mlp = Mlp::load(&vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    cfg: Config,
}

impl Llama {
    /// Forward pass of Llama model, using
    /// flash attention kernels, with paged attention
    /// batching optimizations.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[1, num_total_tokens]`,
    ///     where `num_total_tokens = num_prefill_tokens + num_decode_tokens`
    /// * `input_positions` - Input positions tensor of shape `[1, num_total_tokens]`,
    ///     where `num_total_tokens = num_prefill_tokens + num_decode_tokens`.
    ///     it contains all input positions, so that rotary embeddings can be applied correctly
    /// * `selected_token_indices` - Selected token indices tensor of shape `[1, num_decode_tokens]`
    /// * `kv_caches` - KV caches with paged block arrangement for each model layer. Each tensor is of
    ///      shape `[num_blocks, block_size, num_heads, head_dim]`
    /// * `attention_metadata` - Flash attention metadata, that
    pub fn forward(
        &mut self,
        x: &Tensor,
        input_positions: &Tensor,
        selected_token_indices: &Tensor,
        kv_caches: &Vec<&mut Tensor>,
        attention_metadata: FlashAttentionMetadata,
    ) -> Result<Tensor> {
        if x.dims()[0] != 1 {
            candle_core::bail!(
                "x must be of shape [1, num_total_tokens], got {:?}",
                x.dims()
            );
        }
        let mut x = self.wte.forward(x)?;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            x = block.forward(&x, input_positions, &kv_caches[i], &attention_metadata)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.index_select(selected_token_indices, 1)?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(&format!("model.layers.{i}")), cfg, dtype, device).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            cfg: cfg.clone(),
        })
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionDecodingMetadata, FlashAttentionPrefillMetadata};
    use candle_core::IndexOp;
    use candle_transformers::generation::{LogitsProcessor, Sampling};
    use hf_hub::{api::sync::Api, Repo, RepoType};
    use serial_test::serial;
    use std::io::Write;
    use tokenizers::Tokenizer;

    const EOS_TOKEN: &str = "</s>";

    #[test]
    #[serial]
    fn test_llama_model() -> Result<()> {
        let prompt = "The capital of France is ".to_string();

        let dtype = DType::BF16;
        let device = Device::new_cuda(0).unwrap();
        let model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string();
        let revision = "main".to_string();
        let api = Api::new().expect("Failed to create the HF API");

        println!("loading the model weights from {model_id}");
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api
            .get("tokenizer.json")
            .expect("Failed to get tokenizer.json");
        let config_filename = api.get("config.json").expect("Failed to get config.json");
        let config: LlamaConfig = serde_json::from_slice(
            &std::fs::read(config_filename).expect("Failed to read config.json"),
        )
        .expect("Failed to deserialize config.json");
        let config = config.into_config();

        let filenames = vec![api
            .get("model.safetensors")
            .expect("Failed to get model.safetensors")];
        let mut llama_model = {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            Llama::load(vb, &config, dtype, &device).expect("Failed to load the model")
        };
        let tokenizer =
            Tokenizer::from_file(tokenizer_filename).expect("Failed to load the tokenizer");
        let eos_token_id = config
            .eos_token_id
            .or_else(|| tokenizer.token_to_id(EOS_TOKEN));

        let mut tokens = tokenizer
            .encode(prompt.clone(), true)
            .expect("Failed to encode the prompt")
            .get_ids()
            .to_vec();

        let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);
        println!("starting the inference loop");
        print!("{prompt}");

        let mut logits_processor = {
            let temperature = 0.8;
            let sampling = Sampling::All { temperature };
            LogitsProcessor::from_sampling(42, sampling)
        };

        let sample_len = 32;
        let start_gen = std::time::Instant::now();
        let mut token_generated = 0;

        // kv cache
        let num_blocks = 100;
        let block_size = 16;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let mut kv_cache = std::iter::repeat_with(|| {
            Tensor::zeros(
                (2, num_blocks, block_size, num_key_value_heads, head_dim),
                dtype,
                &device,
            )
        })
        .take(config.num_hidden_layers)
        .collect::<Result<Vec<_>>>()?;

        let kv_cache = kv_cache.iter_mut().collect();

        // prefill forward pass
        let input_positions = Tensor::arange(0, tokens.len() as i64, &device)?.unsqueeze(0)?;
        let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
        let attention_metadata = FlashAttentionMetadata {
            context_lengths: Some(Tensor::from_vec(vec![tokens.len() as u32], (1,), &device)?),
            slot_mapping: Tensor::arange(0, tokens.len() as i64, &device)?,
            decoding_metadata: None,
            num_prefill_tokens: tokens.len(),
            num_decoding_tokens: 0,
            prefill_metadata: Some(FlashAttentionPrefillMetadata {
                block_tables: None,
                max_query_length: Some(tokens.len()),
                max_prefill_sequence_length: tokens.len(),
                query_start_locations: Some(Tensor::from_vec(
                    vec![0, tokens.len() as u32],
                    (2,),
                    &device,
                )?),
                sequence_start_locations: Some(Tensor::from_vec(
                    vec![0, tokens.len() as u32],
                    (2,),
                    &device,
                )?),
                sequence_lengths: Some(Tensor::from_vec(vec![tokens.len() as u32], (1,), &device)?),
            }),
        };
        let logits = llama_model.forward(
            &input,
            &input_positions,
            &Tensor::new(vec![tokens.len() as u32 - 1], &device)?,
            &kv_cache,
            attention_metadata,
        )?;
        let logits = logits.squeeze(0)?.squeeze(0)?;

        let mut next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        // decoding loop
        for _ in 1..sample_len {
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let input_positions = Tensor::new(&[tokens.len() as i64 - 1], &device)?.unsqueeze(0)?;
            let selected_token_indices = Tensor::new(&[0u32], &device)?;
            let num_blocks = (tokens.len() / block_size) as i64 + 1;
            let attention_metadata = FlashAttentionMetadata {
                context_lengths: None,
                slot_mapping: Tensor::new(&[tokens.len() as i64 - 1], &device)?,
                decoding_metadata: Some(FlashAttentionDecodingMetadata {
                    block_tables: Some(
                        Tensor::arange(0, num_blocks, &device)?
                            .reshape((1, num_blocks as usize))?,
                    ),
                    max_decoding_sequence_length: tokens.len(),
                    sequence_lengths: Some(Tensor::new(&[tokens.len() as u32], &device)?),
                }),
                prefill_metadata: None,
                num_prefill_tokens: 0,
                num_decoding_tokens: 1,
            };
            let logits = llama_model
                .forward(
                    &input,
                    &input_positions,
                    &selected_token_indices,
                    &kv_cache,
                    attention_metadata,
                )?
                .squeeze(0)?
                .squeeze(0)?;

            next_token = logits_processor.sample(&logits)?;
            token_generated += 1;
            tokens.push(next_token);

            if Some(next_token) == eos_token_id {
                break;
            }
            if let Some(t) = tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }

        if let Some(rest) = tokenizer.decode_rest().unwrap() {
            print!("{rest}");
        }

        let dt = start_gen.elapsed();
        println!(
            "\n\n{} tokens generated ({} token/s)\n",
            token_generated,
            (token_generated - 1) as f64 / dt.as_secs_f64(),
        );

        Ok(())
    }

    #[test]
    #[serial]
    fn test_llama_model_batch() -> Result<()> {
        let prompts = vec![
            "The capital of France is ".to_string(),
            "Modern music is especially focused on ".to_string(),
            "How many countries do exist ? ".to_string(),
            "Sailing requires advanced techniques on ".to_string(),
            "What are the best places to surf ? ".to_string(),
            "How many letters does the word 'Algarve' has ? ".to_string(),
            "Zero knowledge cryptography regards ".to_string(),
            "What is a large language model ? ".to_string(),
            "What is the best way to learn a new language ? ".to_string(),
            "Healthy food is vital for ".to_string(),
        ];

        let dtype = DType::BF16;
        let device = Device::new_cuda(0).unwrap();
        let model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string();
        let revision = "main".to_string();
        let api = Api::new().expect("Failed to create the HF API");

        println!("loading the model weights from {model_id}");
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api
            .get("tokenizer.json")
            .expect("Failed to get tokenizer.json");
        let config_filename = api.get("config.json").expect("Failed to get config.json");
        let config: LlamaConfig = serde_json::from_slice(
            &std::fs::read(config_filename).expect("Failed to read config.json"),
        )
        .expect("Failed to deserialize config.json");
        let config = config.into_config();

        let filenames = vec![api
            .get("model.safetensors")
            .expect("Failed to get model.safetensors")];
        let mut llama_model = {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            Llama::load(vb, &config, dtype, &device).expect("Failed to load the model")
        };
        let tokenizer =
            Tokenizer::from_file(tokenizer_filename).expect("Failed to load the tokenizer");
        let eos_token_id = config
            .eos_token_id
            .or_else(|| tokenizer.token_to_id(EOS_TOKEN));

        let mut tokens = prompts
            .iter()
            .map(|prompt| {
                tokenizer
                    .encode(prompt.clone(), true)
                    .expect("Failed to encode the prompt")
                    .get_ids()
                    .to_vec()
            })
            .collect::<Vec<_>>();

        let mut tokenizers = std::iter::repeat_with(|| {
            candle_examples::token_output_stream::TokenOutputStream::new(tokenizer.clone())
        })
        .take(10)
        .collect::<Vec<_>>();
        println!("starting the inference loop");
        for prompt in prompts.iter() {
            println!("{prompt}");
        }

        let mut logits_processors = {
            let temperature = 0.8;
            let sampling = Sampling::All { temperature };
            std::iter::repeat_with(|| LogitsProcessor::from_sampling(42, sampling.clone()))
                .take(prompts.len())
                .collect::<Vec<_>>()
        };

        let sample_len = 64;
        let start_gen = std::time::Instant::now();
        let mut token_generated = 0;

        // KV cache
        let num_blocks = 100;
        let block_size = 16;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let mut kv_caches = std::iter::repeat_with(|| {
            Tensor::zeros(
                (2, num_blocks, block_size, num_key_value_heads, head_dim),
                dtype,
                &device,
            )
        })
        .take(config.num_hidden_layers)
        .collect::<Result<Vec<_>>>()?;

        let kv_caches: Vec<_> = kv_caches.iter_mut().collect();

        let num_prefill_tokens = tokens.iter().map(|ts| ts.len()).sum::<usize>();
        let max_tokens_len = tokens.iter().map(|ts| ts.len()).max().unwrap();
        let token_size_allocation = max_tokens_len + 64 + 1;

        println!("tokens: {tokens:?}");
        println!(
            "input: {:?}",
            tokens.clone().into_iter().flatten().collect::<Vec<_>>()
        );
        println!(
            "input_positions: {:?}",
            tokens
                .iter()
                .flat_map(|ts| (0..(ts.len() as i64)))
                .collect::<Vec<_>>()
        );

        // prefill forward pass
        let input_positions = Tensor::from_vec(
            tokens
                .iter()
                .flat_map(|ts| (0..(ts.len() as i64)))
                .collect::<Vec<_>>(),
            (1, num_prefill_tokens),
            &device,
        )?;
        let input = Tensor::from_vec(
            tokens.clone().into_iter().flatten().collect(),
            (1, num_prefill_tokens),
            &device,
        )?;
        let sequence_start_locs = {
            let mut result = Vec::with_capacity(tokens.len() + 1);
            result.push(0); // Start with 0
            tokens.iter().fold(0, |acc, x| {
                let sum = acc + x.len() as u32;
                result.push(sum);
                sum
            });
            result
        };
        println!("sequence_start_locs: {sequence_start_locs:?}");
        println!(
            "context_lengths: {:?}",
            tokens.iter().map(|ts| ts.len() as u32).collect::<Vec<_>>()
        );
        println!(
            "slot_mapping: {:?}",
            tokens
                .iter()
                .enumerate()
                .flat_map(|(i, ts)| {
                    ((i * token_size_allocation) as i64)
                        ..((i * token_size_allocation + ts.len()) as i64)
                })
                .collect::<Vec<_>>()
        );
        let context_lengths = Some(Tensor::from_vec(
            tokens.iter().map(|ts| ts.len() as u32).collect(),
            (tokens.len(),),
            &device,
        )?);
        let slot_mapping = Tensor::from_vec(
            tokens
                .iter()
                .enumerate()
                .flat_map(|(i, ts)| {
                    ((i * token_size_allocation) as i64)
                        ..((i * token_size_allocation + ts.len()) as i64)
                })
                .collect(),
            (num_prefill_tokens,),
            &device,
        )?;
        let query_start_locations = Some(Tensor::from_vec(
            sequence_start_locs.clone(),
            (tokens.len() + 1,),
            &device,
        )?);
        let sequence_start_locations = Some(Tensor::from_vec(
            sequence_start_locs,
            (tokens.len() + 1,),
            &device,
        )?);
        let sequence_lengths = Some(Tensor::from_vec(
            tokens.iter().map(|ts| ts.len() as u32).collect(),
            (tokens.len(),),
            &device,
        )?);
        let attention_metadata = FlashAttentionMetadata {
            context_lengths,
            slot_mapping,
            decoding_metadata: None,
            num_prefill_tokens,
            num_decoding_tokens: 0,
            prefill_metadata: Some(FlashAttentionPrefillMetadata {
                block_tables: None,
                max_query_length: Some(max_tokens_len),
                max_prefill_sequence_length: max_tokens_len,
                query_start_locations,
                sequence_start_locations,
                sequence_lengths,
            }),
        };

        let selected_token_indices = {
            let mut result = Vec::with_capacity(tokens.len());
            let mut i = 0;
            tokens.iter().fold(0, |acc, x| {
                let sum = if i == 0 {
                    i += 1;
                    acc + x.len() as u32 - 1
                } else {
                    acc + x.len() as u32
                };
                result.push(sum);
                sum
            });
            result
        };
        println!("selected_token_indices: {:?}", selected_token_indices);
        let selected_token_indices =
            Tensor::from_vec(selected_token_indices, (tokens.len(),), &device)?;
        let logits = llama_model
            .forward(
                &input,
                &input_positions,
                &selected_token_indices,
                &kv_caches,
                attention_metadata,
            )?
            .squeeze(0)?;

        assert_eq!(logits.dims().len(), 2);
        assert_eq!(logits.dims()[0], 10);
        assert_eq!(logits.dims()[1], 32_000);

        let mut sentences = prompts.clone();

        (0..10).for_each(|i| {
            let next_token = logits_processors[i].sample(&logits.i(i).unwrap()).unwrap();
            if let Some(t) = tokenizers[i].next_token(next_token).unwrap() {
                sentences[i].push_str(&t);
            }
            tokens[i].push(next_token);
        });
        token_generated += 10;

        let mut next_tokens = tokens
            .iter()
            .map(|ts| *ts.last().unwrap())
            .collect::<Vec<_>>();

        println!("next_tokens: {:?}", next_tokens);

        // round division
        let total_num_blocks_per_sequence =
            ((token_size_allocation + block_size - 1) / block_size) as i64;

        let mut finished_sequences = Vec::with_capacity(10);
        let mut active_indices: Vec<usize> = (0..10).collect();

        // decoding loop
        for _ in 1..sample_len {
            println!("active_indices: {active_indices:?}");
            let num_active = active_indices.len();
            if num_active == 0 {
                break; // All sequences have finished
            }

            let input = Tensor::from_vec(
                active_indices
                    .iter()
                    .map(|&i| *tokens[i].last().unwrap())
                    .collect(),
                (1, num_active),
                &device,
            )?;
            let input_positions = Tensor::from_vec(
                active_indices
                    .iter()
                    .map(|&i| tokens[i].len() as i64 - 1)
                    .collect(),
                (1, num_active),
                &device,
            )?;
            let selected_token_indices =
                Tensor::from_vec((0..num_active as u32).collect(), (num_active,), &device)?;
            let max_decoding_sequence_length = active_indices
                .iter()
                .map(|i| tokens[*i].len())
                .max()
                .unwrap();
            let num_blocks_per_sequence = active_indices
                .iter()
                .map(|i| (tokens[*i].len() / block_size) as i64 + 1)
                .collect::<Vec<_>>();
            let max_num_blocks = *num_blocks_per_sequence.iter().max().unwrap() as usize;

            let slot_mapping = Tensor::from_vec(
                active_indices
                    .iter()
                    .map(|&i| (i * token_size_allocation + tokens[i].len()) as i64 - 1)
                    .collect(),
                (num_active,),
                &device,
            )?;

            println!("num_blocks_per_sequence: {:?}", num_blocks_per_sequence);
            println!(
                "slot_mapping: {:?}",
                active_indices
                    .iter()
                    .map(|&i| (i * token_size_allocation + tokens[i].len()) as i64 - 1)
                    .collect::<Vec<_>>()
            );

            println!("total_num_blocks_per_sequence: {total_num_blocks_per_sequence}");
            let block_tables = active_indices
                .iter()
                .zip(num_blocks_per_sequence.iter())
                .flat_map(|(i, num_blocks)| {
                    let mut range = ((*i as i64 * total_num_blocks_per_sequence)
                        ..((*i as i64 * total_num_blocks_per_sequence) + *num_blocks as i64))
                        .collect::<Vec<_>>();
                    range.extend([0i64].repeat(max_num_blocks - *num_blocks as usize)); // pad to max_num_blocks
                    range
                });
            println!(
                "block_tables: {:?}",
                block_tables.clone().collect::<Vec<_>>()
            );
            let block_tables = Some(Tensor::from_vec(
                block_tables.collect(),
                (active_indices.len(), max_num_blocks),
                &device,
            )?);
            let sequence_lengths = Some(Tensor::from_vec(
                active_indices
                    .iter()
                    .map(|&i| tokens[i].len() as u32)
                    .collect::<Vec<_>>(),
                (active_indices.len(),),
                &device,
            )?);

            let attention_metadata = FlashAttentionMetadata {
                context_lengths: None,
                slot_mapping,
                decoding_metadata: Some(FlashAttentionDecodingMetadata {
                    block_tables,
                    max_decoding_sequence_length,
                    sequence_lengths,
                }),
                prefill_metadata: None,
                num_prefill_tokens: 0,
                num_decoding_tokens: num_active,
            };
            let logits = llama_model
                .forward(
                    &input,
                    &input_positions,
                    &selected_token_indices,
                    &kv_caches,
                    attention_metadata,
                )?
                .squeeze(0)?;

            let mut new_active_indices = Vec::new();
            next_tokens = Vec::new();
            for (idx, &i) in active_indices.iter().enumerate() {
                let next_token = logits_processors[i]
                    .sample(&logits.i(idx).unwrap())
                    .unwrap();
                if let Some(t) = tokenizers[i].next_token(next_token).unwrap() {
                    sentences[i].push_str(&t);
                }

                tokens[i].push(next_token);
                next_tokens.push(next_token);

                if Some(next_token) != eos_token_id {
                    new_active_indices.push(i);
                } else {
                    println!("sequence finished for id = {i}");
                    finished_sequences.push(tokens[i].clone());
                }
            }

            active_indices = new_active_indices;
            token_generated += num_active;
        }

        finished_sequences.extend(tokens);

        for i in 0..10 {
            if let Some(rest) = tokenizers[i].decode_rest().unwrap() {
                sentences[i].push_str(&rest);
            }
        }

        let dt = start_gen.elapsed();
        println!(
            "\n\n{} tokens generated ({} token/s)\n",
            token_generated,
            (token_generated - 1) as f64 / dt.as_secs_f64(),
        );

        println!("sentences: {:?}", sentences);

        Ok(())
    }
}
