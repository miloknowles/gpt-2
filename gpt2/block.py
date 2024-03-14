from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2.attention import Attention, MaybeAttention, MaybeKeysValues


class TransformerBlock(nn.Module):
  def __init__(
    self,
    max_position_embeddings: int = 1024,
    embed_dim: int = 768,
    num_heads: int = 8,
    scale: bool = False
  ):
    super(TransformerBlock, self).__init__()

    self.attention = Attention(
      max_position_embeddings=max_position_embeddings,
      embed_dim=embed_dim,
      num_heads=num_heads,
      scale=scale
    )
    self.ln1 = nn.LayerNorm(embed_dim)
    self.ln2 = nn.LayerNorm(embed_dim)
    # Note that in GPT-2, they increase the feature dimension by 4x, then reduce
    # it in a final projection step. After the first FC layer, they apply GELU.
    self.mlp = nn.Sequential(
      nn.Linear(embed_dim, 4*embed_dim),
      # https://paperswithcode.com/method/gelu
      nn.GELU(),
      nn.Linear(4*embed_dim, embed_dim)      
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    past_keys_and_values: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    return_attention: bool = False,
  ) -> tuple[torch.Tensor, MaybeKeysValues, MaybeAttention]:
    """Forward pass of the Transformer block.

    Note that on AIAYN, they take the norm AFTER each operation, whereas GPT-2
    seems to do it before each one. Not sure how much this matters?

    Parameters
    ----------
    * `hidden_states` : Input features with shape (batch, sequence, features).
    """
    hidden_states = self.ln1(hidden_states)

    attn_outputs: tuple[torch.Tensor, MaybeKeysValues, MaybeAttention] = \
      self.attention(
        hidden_states,
        past_keys_and_values,
        use_cache=use_cache,
        return_attention=return_attention
      )

    hidden_states += attn_outputs[0]

    hidden_states = self.ln2(hidden_states)
    hidden_states += self.mlp(hidden_states)

    # hidden_states, present_keys_and_values, present_attention
    return hidden_states, attn_outputs[1], attn_outputs[2]
