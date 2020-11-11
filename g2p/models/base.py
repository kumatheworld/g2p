import torch
import torch.nn as nn

class Base(nn.Module):
    def __init__(self, src_size, enc_embed_dim, hidden_size,
                 tgt_size, dec_embed_dim):
        super().__init__()
        self.enc_emb = nn.Embedding(src_size, enc_embed_dim, padding_idx=0)
        self.dec_emb = nn.Embedding(tgt_size, dec_embed_dim, padding_idx=0)

        self.dec_unembed = nn.Linear(hidden_size, tgt_size - 2)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _prepend_2zeros(out):
        zeros = torch.zeros(*out.size()[:-1], 2,
                            dtype=out.dtype, device=out.device)
        return torch.cat((zeros, out), dim=-1)
