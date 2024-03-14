from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2.exceptions import IncompatibleArguments


MaybeKeysValues = None | tuple[torch.Tensor, torch.Tensor]
MaybeAttention = None | torch.Tensor


class Attention(nn.Module):
  def __init__(
    self,
    max_position_embeddings: int = 1024,
    embed_dim: int = 768,
    num_heads: int = 8,
    scale: bool = False,
    attn_pdrop: float = 0.1,
    resid_pdrop: float = 0.1,
  ):
    """Implements the QKV multiheaded attention mechanism, as used in GPT-2 and similar.

    Parameters
    ----------
    * `max_position_embeddings` : The length of the context window.
    * `embed_dim` : The input feature dimension. This is the dimension of the residual stream.
    * `num_heads`: The number of attention heads. The `embed_dim` must be divisible by this.
    * `scale` : Whether to scale attention  dot products by the inverse square root.
    * `attn_pdrop` : The dropout probability applied to attention maps.
    * `resid_prdrop` : The dropout probability applied to final attended features.
    
    References
    ----------
    https://github.com/openai/gpt-2/blob/master/src/model.py
    """
    super(Attention, self).__init__()

    if (embed_dim % num_heads != 0):
      raise IncompatibleArguments("The `embed_dim` must be divisible by `num_heads`. This is because of how GPT-2 calculates the QKV dimensions for multihead attention.")

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.scale = scale

    # Precompute the attention mask, up to the maximum sequence length.
    # By registering a buffer, this tensor will be moved to whatever device the
    # module is moved to AND will be serialized in the state dict.
    self.register_buffer(
      "attention_mask",
      torch.tril(torch.ones(max_position_embeddings, max_position_embeddings))
    )

    self.W_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
    self.W_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    self.attn_dropout = nn.Dropout(p=attn_pdrop)
    self.resid_dropout = nn.Dropout(p=resid_pdrop)
  
  def _split_heads(self, x: torch.Tensor, num_heads: int, head_embed_dim: int) -> torch.Tensor:
    """Split a tensor of shape (batch, sequence, features) into shape (batch, heads, sequence, head_features)."""
    x = x.reshape((x.shape[0], x.shape[1], num_heads, head_embed_dim))
    return torch.permute(x, (0, 2, 1, 3))

  def _merge_heads(self, x: torch.Tensor):
    """Merge a tensor of shape (batch, heads, sequence, head_features) into shape (batch, sequence, features)."""
    x = x.permute(0, 2, 1, 3) # (B, 1|S, H, f)

    # Concatencate the little features from each head. The reshape squashes the
    # last two dimensions (H, f) into one dimension of size H * f.
    return torch.reshape(x, (x.shape[0], x.shape[1], self.embed_dim))

  def forward(
    self,
    hidden_states: torch.Tensor,
    past_keys_and_values: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    return_attention: bool = False,
  ) -> tuple[torch.Tensor, MaybeKeysValues, MaybeAttention]:
    """Forward pass of QKV attention.

    Parameters
    ----------
    * `hidden_states` :
      The input features with shape (batch, sequence, features).
    * `past_keys_and_values` :
      Precomputed keys and values for past tokens that can be used to speed up inference.
    * `use_cache` : Whether to return the keys and values from this module.
    * `return_attention` : Whether to return the internal attention maps for each head.

    Returns
    -------
    1. The output features with shape (batch, sequence, features).
    2. The computed keys and values from this module (optional).
    3. The attention maps for each head (optional).
    """
    if past_keys_and_values is not None:
      assert(len(past_keys_and_values) == 2) # Expect two entries in the tuple.

    qkv: torch.Tensor = self.W_qkv(hidden_states) # (B, S, 3F)

    q, k, v = qkv.split(self.embed_dim, dim=-1) # Each has dimension (B, S, F).

    # Further split the queries, keys, and values into the slices belonging to each head.
    q = self._split_heads(q, self.num_heads, self.embed_dim // self.num_heads) # B, H, 1|S, f = F // H)
    k = self._split_heads(k, self.num_heads, self.embed_dim // self.num_heads) # B, H, 1|S, f = F // H)
    v = self._split_heads(v, self.num_heads, self.embed_dim // self.num_heads) # B, H, 1|S, f = F // H)

    # If past keys and values (for previous tokens) are passed in, we concatenate
    # the past keys and values with the present ones.
    if past_keys_and_values is not None:
      past_keys, past_values = past_keys_and_values
      k = torch.cat([past_keys, k], dim=-2)
      v = torch.cat([past_values, v], dim=-2)

    # Note that Q * K^T is the same as taking the dot product of every row in
    # Q and and every row in K. Note that 
    w = torch.matmul(q, k.transpose(-2, -1)) # Produces shape (B, H, 1|S, S)

    # As in the original AIAYN paper, scale the dot products of keys and queries
    # by the inverse square root of the key/query dimension. This prevents very
    # large-magnitude dot products from creating small softmax gradients.
    if self.scale:
      w *= 1.0 / math.sqrt(k.size(-1))

    # Mask the attention map so that a token at position `i` can only attend to
    # the tokens at position `<= i`. Then take softmax. By subtracting a large
    # number before the softmax, we effectively zero out the masked elements.
    inf = torch.finfo(w.dtype).max
    # Slice the mask so that it matches the length of the sequence passed in.
    mask = self.attention_mask[:w.size(-2),:w.size(-1)]
    w = F.softmax(w * mask - inf*(1 - mask), dim=-1) # (B, H, 1|S, S)

    # Dropout #1: Apply to the attention map.
    w = self.attn_dropout(w)

    # Compute an attention-weighted sum of values.
    a = torch.matmul(w, v) # (B, H, 1|S, f)

    a = self._merge_heads(a)

    # Project the attended features back to the model's residual stream dimension.
    a = self.W_proj(a) # (B, 1|S, F)
    
    # Dropout #2: Apply to the final attended features.
    a = self.resid_dropout(a)

    # If doing inference with caching, we want to return the present keys and
    # values that have those of the most recent token included.
    return (
      a,
      (k, v) if use_cache else None,
      w if return_attention else None,
    )