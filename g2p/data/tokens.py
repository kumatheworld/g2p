import string
import csv
import torch
import cmudict

class DoubleBets:
    class SingleBet:
        def __init__(self, token_list):
            self.pad_idx, self.st_idx, self.ed_idx = range(3)
            self.special_tokens = ['<pad>', '<st>', '<ed>']
            self.i2t = self.special_tokens + token_list
            self.t2i = {t: i for i, t in enumerate(self.i2t)}

        def __len__(self):
            return len(self.i2t)

        def iseq2tseq(self, iseq):
            return [self.i2t[i] for i in iseq]

        def tseq2iseq(self, tseq, wrap_st=True, wrap_ed=True):
            return torch.LongTensor(
                ([self.st_idx] if wrap_st else []) \
                + ([self.t2i[t] for t in tseq]) \
                + [self.ed_idx] if wrap_ed else []
            )

        def unwrap_iseq(self, iseq):
            return [i for i in iseq if i > self.ed_idx]

    alphabet = SingleBet(list("'-." + string.ascii_lowercase))
    arpabet = SingleBet(cmudict.symbols())

    with open('g2p/data/arpabet2ipa.csv') as f:
        arpabet2ipa = dict(csv.reader(f, delimiter='\t'))

    @classmethod
    def arpabetSeq2ipaString(cls, arpabet_seq):
        return ''.join(cls.arpabet2ipa[arpa] for arpa in arpabet_seq)
