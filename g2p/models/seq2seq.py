import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, src_size, embed_size, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(src_size, embed_size, padding_idx=0)
            self.rnn = nn.RNN(embed_size, hidden_size)

        def forward(self, src_seq):
            x0 = self.embedding(src_seq)
            h0 = torch.zeros(1, x0.size(1), self.hidden_size,
                             device=src_seq.device)
            _, h = self.rnn(x0, h0)
            return h

    class Decoder(nn.Module):
        def __init__(self, embed_size, hidden_size, tgt_size):
            super().__init__()
            self.tgt_size = tgt_size
            self.embedding = nn.Embedding(tgt_size, embed_size, padding_idx=0)
            self.rnn = nn.RNN(embed_size, hidden_size)
            self.unembedding = nn.Linear(hidden_size, tgt_size)
            self.loss_func = nn.CrossEntropyLoss(ignore_index=0)

        def forward(self, h, tgt_seq, search_algo):
            if self.training:
                x = self.embedding(tgt_seq)
                rnn_output_seq, _ = self.rnn(x, h)
                final_output_seq = self.unembedding(rnn_output_seq)
                tgt_seq_label = torch.zeros_like(tgt_seq)
                tgt_seq_label[:-1] = tgt_seq[1:]
                return self.loss_func(
                    final_output_seq.view(-1, final_output_seq.size(-1)),
                    tgt_seq_label.flatten()
                )
            else:
                #TODO: use search algo to make prediction
                return None

    def __init__(self, src_size, enc_embed_size,
                 hidden_size, dec_embed_size, tgt_size):
        super().__init__()
        self.encoder = self.Encoder(src_size, enc_embed_size, hidden_size)
        self.decoder = self.Decoder(dec_embed_size, hidden_size, tgt_size)

    def forward(self, src_seq, tgt_seq=None, search_algo=None):
        h = self.encoder(src_seq)
        return self.decoder(h, tgt_seq, search_algo)
