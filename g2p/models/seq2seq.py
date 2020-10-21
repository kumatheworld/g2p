import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Seq2Seq(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, rnn_type, src_size, embed_size,
                     hidden_size, num_layers, dropout, bidirectional):
            super().__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(src_size, embed_size, padding_idx=0)
            self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size,
                                             num_layers=num_layers,
                                             dropout=dropout,
                                             bidirectional=bidirectional)
            self.hidden_size0 = num_layers * 2 if bidirectional else num_layers

        def forward(self, src_seq_and_len):
            src_seq, src_len = src_seq_and_len
            embedded = self.embedding(src_seq)
            x0 = pack_padded_sequence(embedded, src_len, enforce_sorted=False)
            _, h = self.rnn(x0)
            return h

    class Enc2Dec(nn.Module):
        def __init__(self, num_layers, bidirectional):
            super().__init__()
            if bidirectional:
                class Add2Directions(nn.Module):
                    def __init__(self):
                        super().__init__()

                    def forward(self, h):
                        return h[:num_layers] + h[num_layers:]

                self.enc2dec = Add2Directions()
            else:
                self.enc2dec = nn.Identity()

        def forward(self, hidden):
            return self.enc2dec(hidden)

    class Decoder(nn.Module):
        def __init__(self, rnn_type, embed_size, hidden_size,
                     tgt_size, num_layers, dropout):
            super().__init__()
            self.tgt_size = tgt_size
            self.embedding = nn.Embedding(tgt_size, embed_size, padding_idx=0)
            self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size,
                                             dropout=dropout,
                                             num_layers=num_layers)
            self.unembedding = nn.Linear(hidden_size, tgt_size - 2)
            self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
            self.softmax = nn.Softmax(dim=-1)

        def _compute_logits(self, seq, length, h0):
            embedded = self.embedding(seq)
            x0 = pack_padded_sequence(embedded, length, enforce_sorted=False)
            x, h = self.rnn(x0, h0)
            padded, _ = pad_packed_sequence(x)
            logits = self.unembedding(padded)
            return logits, h

        @staticmethod
        def _prepend_2zeros(out):
            zeros = torch.zeros(*out.size()[:-1], 2,
                                dtype=out.dtype, device=out.device)
            return torch.cat((zeros, out), dim=-1)

        def _rec_prob_gen(self, idx):
            seq = torch.tensor([[idx]], dtype=torch.long,
                               device=self.hidden.device)
            length = torch.ones(1, dtype=torch.long, device=self.hidden.device)
            logits, self.hidden = self._compute_logits(seq, length, self.hidden)
            probabilities = self.softmax(logits)
            return self._prepend_2zeros(probabilities).squeeze(0)

        def forward(self, h, tgt_seq_and_len, search_algo):
            if self.training:
                tgt_seq, tgt_len = tgt_seq_and_len
                logits, _ = self._compute_logits(tgt_seq[:-1], tgt_len, h)
                logits_full = self._prepend_2zeros(logits)
                return self.loss_func(
                    logits_full.view(-1, logits_full.size(-1)),
                    tgt_seq[1:].flatten()
                )
            else:
                predictions = []
                for i in range(h.size(1)):
                    self.hidden = h[:, i:i+1]
                    predictions.append(search_algo(self._rec_prob_gen))
                return predictions


    def __init__(self, rnn_type, src_size, enc_embed_size, hidden_size,
                 dec_embed_size, tgt_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.encoder = self.Encoder(rnn_type, src_size, enc_embed_size,
                                    hidden_size, num_layers, dropout,
                                    bidirectional)
        self.enc2dec = self.Enc2Dec(num_layers, bidirectional)
        self.decoder = self.Decoder(rnn_type, dec_embed_size, hidden_size,
                                    tgt_size, num_layers, dropout)

    def forward(self, data, label=None, search_algo=None):
        return self.decoder(self.enc2dec(self.encoder(data)),
                            label, search_algo)
