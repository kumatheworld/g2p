import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from g2p.models.base import Base

class Seq2Seq(Base):
    def __init__(self, src_size, tgt_size, rnn_type, enc_embed_dim, hidden_size,
                 dec_embed_dim, num_layers, dropout, bidirectional):
        super().__init__(src_size, enc_embed_dim, hidden_size,
                         tgt_size, dec_embed_dim)

        RNN = getattr(nn, rnn_type)

        self.enc_rnn = RNN(enc_embed_dim, hidden_size, num_layers=num_layers,
                           dropout=dropout, bidirectional=bidirectional)

        self.enc2dec = (lambda h: h[:num_layers] + h[num_layers:]) \
                       if bidirectional else nn.Identity()

        self.dec_rnn = RNN(dec_embed_dim, hidden_size, num_layers=num_layers,
                           dropout=dropout)

    def _compute_logits(self, seq, length, h0):
        embedded = self.dec_emb(seq)
        x0 = pack_padded_sequence(embedded, length, enforce_sorted=False)
        x, h = self.dec_rnn(x0, h0)
        padded, _ = pad_packed_sequence(x)
        logits = self.dec_unembed(padded)
        return logits, h

    def _rec_prob_gen(self, idx, h):
        seq = torch.tensor([[idx]], dtype=torch.long, device=h.device)
        length = torch.ones(1, dtype=torch.long)
        logits, h_next = self._compute_logits(seq, length, h)
        prob = self.softmax(logits)
        return self._prepend_2zeros(prob).squeeze(), h_next

    def forward(self, data, label=None, search_algo=None):
        # encoder
        embedded = self.enc_emb(data.seq)
        x0 = pack_padded_sequence(embedded, data.len, enforce_sorted=False)
        _, h = self.enc_rnn(x0)

        # enc2dec
        h = self.enc2dec(h)

        # decoder
        if self.training:
            logits, _ = self._compute_logits(label.seq[:-1], label.len - 1, h)
            logits_full = self._prepend_2zeros(logits)
            return self.loss_func(
                logits_full.view(-1, logits_full.size(-1)),
                label.seq[1:].flatten()
            )
        else:
            return [
                search_algo(self._rec_prob_gen, h[:, [i]])
                for i in range(h.size(1))
            ]
