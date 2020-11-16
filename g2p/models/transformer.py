import math
import torch
import torch.nn as nn
from g2p.models.base import Base

class Transformer(Base):
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=50):
            super().__init__()
            self.d_model = d_model
            pe = torch.empty(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float()
                                 * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            x = x * math.sqrt(self.d_model) + self.pe[:x.size(0)]
            return self.dropout(x)

    def __init__(self, src_size, tgt_size, d_model, nhead, dim_feedforward,
                 dropout, num_layers):
        super().__init__(src_size, d_model, d_model, tgt_size, d_model)

        self.pos_enc = self.PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                   dim_feedforward, dropout)
        self.enc_trf = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead,
                                                   dim_feedforward, dropout)
        self.dec_trf = nn.TransformerDecoder(decoder_layer, num_layers)

    def _compute_logits(self, seq):
        embedded = self.dec_emb(seq)
        trues = torch.ones(seq.size(0), seq.size(0),
                           dtype=torch.bool, device=seq.device)
        tgt_mask = torch.triu(trues, diagonal=1)
        transformed = self.dec_trf(embedded, self.memory, tgt_mask,
                                   tgt_key_padding_mask=self.tkpm,
                                   memory_key_padding_mask=self.skpm)
        logits = self.dec_unembed(transformed)
        return logits

    def _rec_prob_gen(self, idx, seq):
        idx_ = torch.tensor([[idx]], dtype=torch.long, device=seq.device)
        seq_ext = torch.cat((seq, idx_))
        logits = self._compute_logits(seq_ext)[[seq.size(0)]]
        prob = self.softmax(logits)
        return self._prepend_2zeros(prob).squeeze(), seq_ext

    def forward(self, data, label=None, search_algo=None):
        src_emb = self.pos_enc(self.enc_emb(data.seq))
        skpm = ~data.seq.bool().t()
        memory = self.enc_trf(src_emb, src_key_padding_mask=skpm)

        if self.training:
            self.skpm = skpm
            self.tkpm = ~label.seq[:-1].bool().t()
            self.memory = memory
            logits = self._compute_logits(label.seq[:-1])
            logits_full = self._prepend_2zeros(logits)
            return self.loss_func(
                logits_full.view(-1, logits_full.size(-1)),
                label.seq[1:].flatten()
            )
        else:
            self.tkpm = None
            seq_empty = torch.empty(0, 1, dtype=torch.long,
                                    device=data.seq.device)
            results = []
            for i, skpm_ in enumerate(skpm):
                self.skpm = skpm_.unsqueeze(0)
                self.memory = memory[:, [i]]
                results.append(search_algo(self._rec_prob_gen, seq_empty))
            return results
