import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2.block import TransformerBlock
from gpt2.attention import MaybeKeysValues, MaybeAttention


class GPT2(nn.Module):
  def __init__(
    self,
    vocab_size: int = 30_000,
    max_position_embeddings: int = 1024,
    embed_dim: int = 768,
    num_heads: int = 8,
    num_hidden_layers: int = 12,
    scale: bool = False
  ):
    """Implements GPT-2 based on the original OpenAI code.
    
    References
    ----------
    https://github.com/openai/gpt-2/blob/master/src/model.py
    """
    self.token_embedding = nn.Embedding(vocab_size, embed_dim) # wte
    self.position_embedding = nn.Embedding(max_position_embeddings, embed_dim) # wpe

    self.blocks = nn.ModuleList(*[TransformerBlock(
      max_position_embeddings=max_position_embeddings,
      embed_dim=embed_dim,
      num_heads=num_heads,
      scale=scale
    ) for _ in range(num_hidden_layers)])

    self.final_layer_norm = nn.LayerNorm(embed_dim)

  def forward(
    self,
    input_ids: torch.LongTensor,
    past_key_values: tuple[tuple[MaybeKeysValues]] | None,
  ):
    """Forward pass for a headless GPT-2.

    This module returns the raw hidden state of the model, and must be combined
    with a language model head for training.
    
    Parameters
    ----------
    * `input_ids` :
      The input token ids with shape (batch, sequence). Note that this is NOT a
      one-hot encoding, and contains vocab indices to be more compact.

    Returns
    -------
    The final embeddings of shape (batch, sequence, features).
    """
    hidden_states = self.token_embedding(input_ids)
    hidden_states = hidden_states + self.position_embedding()

    # TODO: dropout!
    #         if token_type_ids is not None:
            # token_type_embeds = self.wte(token_type_ids)
            # hidden_states = hidden_states + token_type_embeds

    if past_key_values is None:
      past_key_values = [None for _ in range(len(self.blocks))]

    for block, past in zip(self.blocks, past_key_values):
      outputs = block(hidden_states, past, return_attention=False)
      hidden_states = outputs[0]

    hidden_states = self.final_layer_norm(hidden_states)
    return hidden_states


class GPT2LMHead(nn.Module):
  def __init__(self, vocab_size: int = 30_000, embed_dim: int = 768):
    """Decodes the raw features of GPT-2 into logits that can be used for next word prediction."""
    self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
  
  def _tie_weights(self, encoder_weights: torch.Tensor):
    """Ensure that the encoded and decoder share weights."""
    self.decoder.weight = encoder_weights.transpose()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Returns the logits for next-word prediction."""
    lm_logits = self.decoder(x)
    return lm_logits