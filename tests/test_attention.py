import torch
import torch.nn as nn
import math

from gpt2.attention import Attention


def test_attention_large():
  with torch.no_grad():
    n_batch, max_position_embeddings, embed_dim, num_heads = 2, 1024, 768, 12
    m = Attention(max_position_embeddings, embed_dim, num_heads, scale=True)
    x = torch.ones(n_batch, max_position_embeddings, embed_dim)
    out = m(x)
    assert(out[0].shape == (n_batch, max_position_embeddings, embed_dim))


def test_attention_identity():
  """Test that attention produces the correct results.
  
  To make the test simpler to reason about, we set the Q, K, and V matrices to
  be the identity. This means that the network computes them by simply taking
  the relevant slice of the input features for each attention head.
  """
  n_batch, max_position_embeddings, embed_dim, num_heads = 1, 3, 4, 2

  m = Attention(
    max_position_embeddings=max_position_embeddings,
    embed_dim=embed_dim,
    num_heads=num_heads,
    scale=False
  )

  with torch.no_grad():
    m.eval()

    # The query, key, and value matrices are the identity; they just return the
    # original features from the residual stream.
    Q, K, V = torch.eye(embed_dim), torch.eye(embed_dim), torch.eye(embed_dim)
    m.W_qkv.weight = nn.Parameter(torch.concat([Q, K, V], dim=0)) # (3F, F)
    m.W_qkv.bias = nn.Parameter(torch.zeros(embed_dim * 3))
    m.W_proj.weight = nn.Parameter(torch.eye(embed_dim))
    m.W_proj.bias = nn.Parameter(torch.zeros(embed_dim))

    # Each row is the embedding for a token in the sequence. We make them
    # orthogonal to reduce the complexity of attention. Because there are 2 heads,
    # the "left" half of the feature subspace will go to the first head, and the
    # "right" half will go to the second head.
    x = torch.Tensor([
      [1, 0,   0, 0],
      [0, 1,   0, 0],
      [0, 0,   1, 0]
    ]).unsqueeze(0)

    assert(x.shape == (n_batch, max_position_embeddings, embed_dim))

    out = m(x)
    
    e = math.e
    e_1 = math.e + 1
    e_2 = math.e + 2

    expected = torch.Tensor([
      [1,     0,     0,     0],
      [1/e_1, e/e_1, 0,     0],
      [1/3,   1/3, e/e_2,   0]
    ])

    torch.testing.assert_close(out[0].squeeze(0), expected)