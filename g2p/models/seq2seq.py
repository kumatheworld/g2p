import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from g2p.models.base import Base

class Seq2Seq(Base):
    def __init__(self, rnn_type, src_size, enc_embed_dim, hidden_size,
                 dec_embed_dim, tgt_size, num_layers, dropout, bidirectional):
        super().__init__(src_size, enc_embed_dim, tgt_size, dec_embed_dim)

        RNN = getattr(nn, rnn_type)

        # encoder
        self.enc_rnn = RNN(enc_embed_dim, hidden_size, num_layers=num_layers,
                           dropout=dropout, bidirectional=bidirectional)

        # decoder
        self.dec_rnn = RNN(dec_embed_dim, hidden_size, num_layers=num_layers,
                           dropout=dropout)
        self.dec_unembed = nn.Linear(hidden_size, tgt_size - 2)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        self.softmax = nn.Softmax(dim=-1)

    def _compute_logits(self, seq, length, h0):
        embedded = self.dec_emb(seq)
        x0 = pack_padded_sequence(embedded, length, enforce_sorted=False)
        x, h = self.dec_rnn(x0, h0)
        padded, _ = pad_packed_sequence(x)
        logits = self.dec_unembed(padded)
        return logits, h

    @staticmethod
    def _prepend_2zeros(out):
        zeros = torch.zeros(*out.size()[:-1], 2,
                            dtype=out.dtype, device=out.device)
        return torch.cat((zeros, out), dim=-1)

    def _rec_prob_gen(self, idx, h):
        seq = torch.tensor([[idx]], dtype=torch.long, device=h.device)
        length = torch.ones(1, dtype=torch.long, device=h.device)
        logits, h_next = self._compute_logits(seq, length, h)
        prob = self.softmax(logits)
        return self._prepend_2zeros(prob).squeeze(), h_next

    def forward(self, data, label=None, search_algo=None):
        # encoder
        embedded = self.enc_emb(data.seq)
        x0 = pack_padded_sequence(embedded, data.len, enforce_sorted=False)
        _, h = self.enc_rnn(x0)

        # enc2dec
        if self.bidirectional:
            h = h[:self.num_layers] + h[self.num_layers:]

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
                search_algo(self._rec_prob_gen, h[:, i:i+1])
                for i in range(h.size(1))
            ]
