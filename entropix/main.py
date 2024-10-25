from typing import Tuple

import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, sample
from entropix.prompts import create_prompts_from_csv, prompt6, p4o
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / '../weights'

def apply_scaling(freqs: jax.Array):
  SCALE_FACTOR = 8
  LOW_FREQ_FACTOR = 1
  HIGH_FREQ_FACTOR = 4
  OLD_CONTEXT_LEN = 8192  # original llama3 length

  low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
  high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

  def scale_freq(freq):
    wavelen = 2 * math.pi / freq

    def scale_mid(_):
      smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
      return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

    return jax.lax.cond(
      wavelen < high_freq_wavelen,
      lambda _: freq,
      lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None),
      None
    )

  return jax.vmap(scale_freq)(freqs)


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
  freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
  if use_scaled:
    freqs = apply_scaling(freqs)
  t = jnp.arange(end, dtype=dtype)
  freqs = jnp.outer(t, freqs)
  return jnp.exp(1j * freqs)


def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
  mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
  if seqlen > 1:
    mask = jnp.full((seqlen, seqlen), float('-inf'))
    mask = jnp.triu(mask, k=1)
    mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
  return mask

#def main(weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1B-Instruct')):
def main(weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('70B-Nemotron-Instruct')):
  model_params = LLAMA_1B_PARAMS
  xfmr_weights, mesh = load_weights(weights_path.absolute(), model_params)
  tokenizer = Tokenizer('entropix/tokenizer.model')
  xfmr_fn = jax.jit(xfmr, static_argnames=("model_params", "decode"))
  sample_fn = jax.jit(sample, static_argnames=("cfg",))

  # Create the batch of tokens
  def generate(xfmr_weights, model_params, tokens):
    gen_tokens = None
    cur_pos = 0
    tokens = jnp.array([tokens], jnp.int32)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    replicated = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    freqs_cis = jax.device_put(precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope), replicated)
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    logits, kvcache, _, _ = xfmr_fn(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
    next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    cur_pos = seqlen
    stop = jax.device_put(jnp.array([128001, 128008, 128009]), replicated)
    sampler_cfg = SamplerConfig()
    gen_tokens = [next_token]
    profile = False
    start_pos = cur_pos

    start = time.time()
    while cur_pos < 8192:
      cur_pos += 1
      if cur_pos == start_pos + 10:
        start = time.time()
      if profile and cur_pos == start_pos + 10:
        jax.profiler.start_trace("profile")
      logits, kvcache, scores, stats = xfmr_fn(xfmr_weights, model_params, next_token, cur_pos, freqs_cis, kvcache, decode=True)
      next_token = sample_fn(logits, scores, cfg=sampler_cfg)
      gen_tokens.append(next_token)
      out_token = tokenizer.decode(next_token.tolist()[0])
      print(out_token, end='', flush=True)
      if profile and cur_pos == start_pos + 10:
        jax.profiler.stop_trace()
      if jnp.isin(next_token, stop).any():
        break

    print(f"{(cur_pos - (start_pos + 10))/(time.time() - start):.3} toks/s")

  prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
 
Think carefully in a step-by-step manner. Oliver picks 44 kiwis on Friday. Then he picks 58 kiwis on Saturday. On Sunday, he picks double the number of kiwis he did on Friday, but five of them were a bit smaller than average. How many kiwis does Oliver have?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
  print(prompt)
  tokens = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
  with mesh:
    generate(xfmr_weights, model_params, tokens)


import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

if __name__ == '__main__':
  tyro.cli(main)
