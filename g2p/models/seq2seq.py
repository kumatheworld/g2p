import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, rnn_type, src_size, embed_size, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(src_size, embed_size, padding_idx=0)
            self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size)

        def forward(self, src_seq):
            x0 = self.embedding(src_seq)
            h0 = torch.zeros(1, x0.size(1), self.hidden_size,
                             device=src_seq.device)
            _, h = self.rnn(x0, h0)
            return h

    class Decoder(nn.Module):
        def __init__(self, rnn_type, embed_size, hidden_size, tgt_size):
            super().__init__()
            self.tgt_size = tgt_size
            self.embedding = nn.Embedding(tgt_size, embed_size, padding_idx=0)
            self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size)
            self.unembedding = nn.Linear(hidden_size, tgt_size - 2)
            self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
            self.softmax = nn.Softmax(dim=0)

        def _forward_common(self, seq_in, h0):
            x0 = self.embedding(seq_in)
            x, h = self.rnn(x0, h0)
            seq_out = self.unembedding(x)
            mask_pad_and_st = torch.full_like(seq_out, float('-inf'))[..., :2]
            logits = torch.cat((mask_pad_and_st, seq_out), dim=-1)
            return logits, h

        def _rec_prob_gen(self, idx):
            seq_in = torch.tensor([[idx]], dtype=torch.long,
                                  device=self.hidden.device)
            logits, self.hidden = self._forward_common(seq_in, self.hidden)
            return self.softmax(logits.squeeze(0))

        def forward(self, h, tgt_seq, search_algo):
            if self.training:
                logits, _ = self._forward_common(tgt_seq, h)
                tgt_seq_label = torch.zeros_like(tgt_seq)
                tgt_seq_label[:-1] = tgt_seq[1:]
                return self.loss_func(logits.view(-1, logits.size(-1)),
                                      tgt_seq_label.flatten())
            else:
                predictions = []
                for i in range(h.size(1)):
                    self.hidden = h[:, i:i+1]
                    predictions.append(search_algo(self._rec_prob_gen))
                return predictions

    def __init__(self, rnn_type, src_size, enc_embed_size,
                 hidden_size, dec_embed_size, tgt_size):
        super().__init__()
        self.encoder = self.Encoder(rnn_type, src_size,
                                    enc_embed_size, hidden_size)
        self.decoder = self.Decoder(rnn_type, dec_embed_size,
                                    hidden_size, tgt_size)

    def forward(self, src_seq, tgt_seq=None, search_algo=None):
        h = self.encoder(src_seq)
        return self.decoder(h, tgt_seq, search_algo)
