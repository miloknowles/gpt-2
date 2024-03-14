import torch
import torch.nn as nn

from gpt2.model import GPT2


n_batch = 2
n_sequence = 10
n_feature = 768
n_heads = 8
n_layers = 12


def test_model_no_past():
  model = GPT2(
    vocab_size=2048,
    max_position_embeddings=1024,
    embed_dim=n_feature,
    num_heads=n_heads,
    num_hidden_layers=n_layers,
    scale=False,
  )

  input_ids = torch.arange(n_sequence).unsqueeze(0).repeat(n_batch, 1).long() # (batch, sequence)

  out = model(input_ids, past_key_values=None, use_cache=False)
  assert(out[0].shape == (n_batch, n_sequence, n_feature))

  out = model(input_ids, past_key_values=None, use_cache=True)
  assert(out[0].shape == (n_batch, n_sequence, n_feature))

  assert(len(out[1]) == n_layers) # One entry stored for each layer.
  assert(len(out[1][0]) == 2) # Should contains keys and values.

  # The cached keys and values should be in their "head-reshaped" format.
  assert(out[1][0][0].shape == (n_batch, n_heads, n_sequence, n_feature // n_heads))
  assert(out[1][0][1].shape == (n_batch, n_heads, n_sequence, n_feature // n_heads))


def test_model_with_past():
  model = GPT2(
    vocab_size=2048,
    max_position_embeddings=1024,
    embed_dim=n_feature,
    num_heads=n_heads,
    num_hidden_layers=n_layers,
    scale=False,
  )

  past_keys_values = ()

  for _ in range(n_layers):
    past_keys = torch.ones(n_batch, n_heads, n_sequence, n_feature // n_heads) # (batch, heads, sequence, head_features)
    past_values = torch.ones_like(past_keys)
    past_keys_values += ((past_keys, past_values),)

  # Now we just pass in the input_ids for the next token only.
  input_ids = torch.arange(1).unsqueeze(0).repeat(n_batch, 1) # (batch, 1)

  out = model(input_ids, past_key_values=past_keys_values, use_cache=True)
  # assert(out[0].shape == (n_batch, n_sequence, n_feature))

  # assert(len(out[1]) == n_layers) # One entry stored for each layer.
  # assert(len(out[1][0]) == 2) # Should contains keys and values.

  # # The cached keys and values should be in their "head-reshaped" format.
  # assert(out[1][0][0].shape == (n_batch, n_heads, n_sequence, n_feature // n_heads))
  # assert(out[1][0][1].shape == (n_batch, n_heads, n_sequence, n_feature // n_heads))