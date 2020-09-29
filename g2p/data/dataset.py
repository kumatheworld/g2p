import cmudict
from . import DoubleBets

a2a_dataset = [
    [DoubleBets.alphabet.tseq2iseq(alphaseq),
     DoubleBets.arpabet.tseq2iseq(arpaseq)]
    for k, vs in cmudict.dict().items()
    for alphaseq, arpaseq in [(k, v) for v in vs]
]
