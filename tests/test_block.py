import torch
import torch.nn as nn

from gpt2.block import TransformerBlock


def test_block_large():
  n_batch, max_position_embeddings, embed_dim, num_heads = 2, 1024, 768, 12
  m = TransformerBlock(max_position_embeddings, embed_dim, num_heads, scale=True)

  x = torch.ones(n_batch, max_position_embeddings, embed_dim)
  out = m(x)
  assert(out[0].shape == (n_batch, max_position_embeddings, embed_dim))