import numpy as np
from g2p.data import DoubleBets

def mean_score(eval_func, pred, label_seq):
    return np.mean([
        eval_func(p, DoubleBets.arpabet.unwrap_iseq(l))
        for p, l in zip(pred, label_seq.t())
    ])
