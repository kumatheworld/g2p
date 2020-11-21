import argparse
import torch

from g2p.config import Config
from g2p.data import Data, DoubleBets

def predict(device, model, search_algo, word):
    input_seq = DoubleBets.alphabet.tseq2iseq(word).unsqueeze(1).to(device)
    input_len = torch.LongTensor([input_seq.size(0)])
    with torch.no_grad():
        data = Data(input_seq, input_len)
        output_seq = model(data, search_algo=search_algo)[0]
        output_arpaseq = DoubleBets.arpabet.iseq2tseq(output_seq)
        return DoubleBets.arpabetSeq2ipaString(output_arpaseq)

def test(cfg):
    model = cfg.MODEL
    model.eval()
    print("Model ready, type whatever word you like!")

    while True:
        word = input()
        prediction = predict(cfg.DEVICE, model, cfg.SEARCH, word)
        print(prediction)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default',
                        help="YAML file name under configs/")
    args = parser.parse_args()

    cfg = Config(args.config, train=False)
    test(cfg)
