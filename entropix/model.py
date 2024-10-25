from typing import Optional, Tuple
import math

import jax
import jax.numpy as jnp

from functools import partial

from entropix.config import ModelParams
from entropix.kvcache import KVCache
from entropix.stats import AttnStats
from entropix.weights import XfmrWeights, LayerWeights
from jax.sharding import PartitionSpec as PS
from jax.experimental.pallas.ops.gpu.rms_norm import rms_norm as pl_rms_norm

shard = jax.lax.with_sharding_constraint

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
USE_PL_RMS_NORM = True
USE_CUDNN_ATTENTION = True

def rms_norm(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
  x = shard(x, PS())
  if USE_PL_RMS_NORM:
    return pl_rms_norm(x, w, jnp.zeros_like(w))
  return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))

def apply_rotary_emb(xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array, dtype: jnp.dtype = jnp.float32) -> Tuple[jax.Array, jax.Array]:
  reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
  reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
  xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
  xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
  xq_out = xq_ * freqs_cis[None, :, None, :]
  xk_out = xk_ * freqs_cis[None, :, None, :]
  xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
  xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
  return xq_out.astype(dtype), xk_out.astype(dtype)

def attention(x: jax.Array, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: jax.Array, kvcache: KVCache, attn_mask: Optional[jax.Array] = None) -> Tuple[jax.Array, KVCache]:
  bsz, _, _ = x.shape
  n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
  xq = jnp.einsum('...e,enh->...nh', x, layer_weights.wq)
  xk = jnp.einsum('...e,enh->...nh', x, layer_weights.wk)
  xv = jnp.einsum('...e,enh->...nh', x, layer_weights.wv)
  xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
  keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
  
  if USE_CUDNN_ATTENTION:
    bf16 = jnp.bfloat16
    assert xq.ndim == keys.ndim == values.ndim == 4
    # check the sharding annotation here
    q = shard(xq.astype(bf16), PS(None, None, "mp", None)) # [B, H, T, D]
    k = shard(keys.astype(bf16), PS(None, None, "mp", None)) # [B, H, S, D]
    v = shard(values.astype(bf16), PS(None, None, "mp", None)) # [B, H, S, D]
    if attn_mask is not None:
      mask = (attn_mask > 0.5 * jnp.finfo(bf16).min)[None, :, :]
      mask = jnp.pad(mask, ((0, 0), (0, 0), (0, keys.shape[1] - xq.shape[1])))
    else:
      assert q.shape[1] == 1
      mask = jnp.swapaxes(jnp.any(k != 0, axis=-1), -2, -1)[:, :, None, :]
      # check the sharding annotation here
      mask = shard(mask, PS(None, "mp", None, None)) # [B, H, T, S]
    output = jax.nn.dot_product_attention(q, k, v, mask=mask, 
                                          scale=float(1.0 / math.sqrt(model_params.head_dim)), 
                                          implementation="cudnn").astype(xq.dtype)
    # perhaps omit this computation for performance
    # but it's being used in sampling
    pre_scores = jnp.einsum('...qnh,...knh->...nqk', xq, keys) / jnp.sqrt(model_params.head_dim)

    output = output.reshape((output.shape[0], output.shape[1], -1))
  else:
    scores = jnp.einsum('...qnh,...knh->...nqk', xq, keys)
    pre_scores = scores / jnp.sqrt(model_params.head_dim)
    scores = pre_scores.astype(jnp.float32)  # Always do attention softmax at float32
    if attn_mask is not None:
      scores = scores.at[..., :attn_mask.shape[-1]].add(attn_mask)
    mask = jnp.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)

    padded_logits = jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    scores = jax.nn.softmax(padded_logits, axis=-1).astype(x.dtype)
    output = jnp.einsum('...nqk,...knh->...qnh', scores, values)
    output = output.reshape((output.shape[0], output.shape[1], -1))

  out = shard(jnp.dot(output, layer_weights.wo), PS())
  return out, kvcache, pre_scores

def feed_forward(x: jax.Array, layer_weights: LayerWeights) -> jax.Array:
 x = shard(x, PS())
 h1 = jax.nn.silu(shard(jnp.dot(x, layer_weights.w1), PS(None, None, 'mp')))
 h =  h1 * shard(jnp.dot(x, layer_weights.w3), PS(None, None, 'mp'))
 return shard(jnp.dot(h, layer_weights.w2), PS())

def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: jax.Array, cur_pos: int, freqs_cis: jax.Array, kvcache: KVCache, attn_mask: Optional[jax.Array]=None, decode=False) -> Tuple[jax.Array, KVCache]:
  if decode:
    freqs_cis = jnp.expand_dims(freqs_cis[cur_pos], 0)
  h = xfmr_weights.tok_embeddings[tokens]
  attn_stats = AttnStats.new(
    bsz=tokens.shape[0],
    n_layers=model_params.n_layers,
    n_heads=model_params.n_local_heads
  )
  for i in range(model_params.n_layers):
    norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
    h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
    attn_stats = attn_stats.update(scores[:,:,-1,:], i)
    h = h + h_attn
    h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
  logits = jnp.dot(rms_norm(h, xfmr_weights.norm), xfmr_weights.output.T)
  return logits, kvcache, scores, attn_stats
