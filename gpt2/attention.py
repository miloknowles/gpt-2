import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
  def __init__(
    self,
    n_sequence: int = 1024,
    d_features: int = 768,
    n_heads: int = 8,
    scale: bool = False
  ):
    """Implements the QKV multiheaded attention mechanism, as used in GPT-2 and similar.
    
    References
    ----------
    https://github.com/openai/gpt-2/blob/master/src/model.py
    """
    super(Attention, self).__init__()

    if (d_features % n_heads != 0):
      raise ValueError("The `d_features` must be divisible by `n_heads`. This is because of how GPT-2 calculates the QKV dimensions for multihead attention.")

    self.d_features = d_features
    self.n_heads = n_heads
    self.scale = scale

    self.attn_mask = torch.tril(torch.ones(n_sequence, n_sequence))

    self.W_qkv = nn.Linear(d_features, 3 * d_features, bias=True)
    self.W_proj = nn.Linear(d_features, d_features, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of QKV attention.
    
    Parameters
    ----------
    `x` : The input features with shape (batch, sequence, features).

    Returns
    -------
    The output features with shape (batch, sequence, features).
    """
    qkv: torch.Tensor = self.W_qkv(x) # (B, S, 3F)

    q, k, v = qkv.split(self.d_features, dim=-1) # Each has dimension (B, S, F).

    # Further split the queries, keys, and values into the slices belonging to each head.
    nb, ns, nf = x.size()
    q = q.reshape((nb, ns, self.n_heads, nf // self.n_heads)) # Now (B, S, H, f = F // H)
    k = q.reshape((nb, ns, self.n_heads, nf // self.n_heads)) # Now (B, S, H, f = F // H)
    v = q.reshape((nb, ns, self.n_heads, nf // self.n_heads)) # Now (B, S, H, f = F // H)

    q = torch.permute(q, (0, 2, 1, 3)) # Want (B, H, S, f)
    k = torch.permute(k, (0, 2, 1, 3)) # Want (B, H, S, f)
    v = torch.permute(v, (0, 2, 1, 3)) # Want (B, H, S, f)

    # Note that Q * K^T is the same as taking the dot product of every row in
    # Q and and every row in K.
    w = torch.matmul(q, k.transpose(-2, -1)) # Produces shape (B, H, S, S)

    # As in the original AIAYN paper, scale the dot products of keys and queries
    # by the inverse square root of the key/query dimension. This prevents very
    # large-magnitude dot products from creating small softmax gradients.
    if self.scale:
      w *= 1.0 / math.sqrt(k.size(-1))

    # Mask the attention map so that a token at position `i` can only attend to
    # the tokens at position `<= i`. Then take softmax. By subtracting a large
    # number before the softmax, we effectively zero out the masked elements.
    w = F.softmax(w * self.attn_mask - 1e10*(1 - self.attn_mask), dim=-1) # (B, H, S, S)

    # Compute an attention-weighted sum of values.
    a = torch.matmul(w, v) # (B, H, S, f)
    a = a.permute(0, 2, 1, 3)

    # Concatencate the little features from each head. The reshape squashes the
    # last two dimensions (H, f) into one dimension of size H * f.
    a = torch.reshape(a, (nb, ns, nf))

    # Project the attended features back to the model's residual stream dimension.
    a = self.W_proj(a) # (B, S, F)

    return a