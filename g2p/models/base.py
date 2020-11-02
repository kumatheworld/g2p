import torch.nn as nn

class Base(nn.Module):
    def __init__(self, src_size, enc_embed_dim, tgt_size, dec_embed_dim):
        super().__init__()
        self.enc_emb = nn.Embedding(src_size, enc_embed_dim, padding_idx=0)
        self.dec_emb = nn.Embedding(tgt_size, dec_embed_dim, padding_idx=0)
