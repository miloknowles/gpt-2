import torch
import torch.nn as nn

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
    super(GPT2, self).__init__()

    self.token_embedding = nn.Embedding(vocab_size, embed_dim) # wte
    self.position_embedding = nn.Embedding(max_position_embeddings, embed_dim) # wpe

    self.blocks = nn.ModuleList([TransformerBlock(
      max_position_embeddings=max_position_embeddings,
      embed_dim=embed_dim,
      num_heads=num_heads,
      scale=scale
    ) for _ in range(num_hidden_layers)])

    self.final_layer_norm = nn.LayerNorm(embed_dim)

  def _encode_positions(self, offset: int, length: int) -> torch.Tensor:
    """Encodings positions, starting from `offset`.
    
    The offset is used when `forward` also receives some cached keys and
    values. The token that it's predicting is not actually at position zero in
    that case, but at the index after the sequence that's already been generated.
    """
    token_positions = torch.arange(offset, offset + length).unsqueeze(0)
    return self.position_embedding(token_positions)

  def forward(
    self,
    input_ids: torch.LongTensor,
    past_key_values: tuple[tuple[MaybeKeysValues]] | None,
    use_cache: bool = False,
  ) -> tuple[torch.Tensor, tuple[MaybeKeysValues] | None]:
    """Forward pass for a headless GPT-2.

    This module returns the raw hidden state of the model, and must be combined
    with a language model head for training.
    
    Parameters
    ----------
    * `input_ids` :
      The input token ids with shape (batch, sequence). Note that this is NOT a
      one-hot encoding, and contains vocab indices to be more compact.
    * `past_key_values` :
      Optional precomputed keys and values for all layers and past token positions.
      There should be an entry for each layer in the network, and each entry should
      be a tuple with two entries, the keys and values. Keys and values have shape
      (batch, heads, sequence, head_embed_dim).
    * `use_cache` : Whether to return the computed keys and values for each layer.

    Returns
    -------
    A tuple with two entries:

    1. The final embeddings of shape (batch, sequence, features).
    2. The computed keys and values for all layers if `use_cache` is on. This is
       a tuple with an entry for each layer. Each entry is a tuple of size 2,
       containing keys and values of shape (batch, heads, sequence, head_features).
    """
    assert(len(input_ids.shape) == 2)

    token_embed = self.token_embedding(input_ids)

    past_length = 0 if past_key_values is None else past_key_values[0][0].size(-2)
    position_embed = self._encode_positions(past_length, input_ids.size(-1))

    hidden_states = token_embed + position_embed

    # TODO: dropout!
    # TODO: token type IDs

    if past_key_values is None:
      past_key_values = [None for _ in range(len(self.blocks))]

    present_keys_and_values = () if use_cache else None

    for block, past_key_values_layer in zip(self.blocks, past_key_values):
      outputs = block(hidden_states, past_key_values_layer, use_cache=use_cache, return_attention=False)
      hidden_states = outputs[0]

      # Append the keys and values for this layer to the stack.
      if use_cache:
        present_keys_and_values = present_keys_and_values + (outputs[1],)

    hidden_states = self.final_layer_norm(hidden_states)

    return hidden_states, present_keys_and_values


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