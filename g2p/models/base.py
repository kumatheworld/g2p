import torch.nn as nn

class Base(nn.Module):
    def __init__(self, src_size, enc_embed_size, tgt_size, dec_embed_size):
        super().__init__()
        self.enc_emb = nn.Embedding(src_size, enc_embed_size, padding_idx=0)
        self.dec_emb = nn.Embedding(tgt_size, dec_embed_size, padding_idx=0)
