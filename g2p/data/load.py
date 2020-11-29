from collections import namedtuple
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

Data = namedtuple('Data', 'seq len')

def _pad_data_and_label(device):
    def _pad_and_get_len(seqs):
        return Data(pad_sequence(seqs).to(device),
               torch.LongTensor([len(seq) for seq in seqs]))

    def _pad_data_and_label_(batch):
        data, label = zip(*batch)
        return _pad_and_get_len(data), _pad_and_get_len(label)

    return _pad_data_and_label_

def get_loader(dataset, batch_size, device, shuffle=True):
    return DataLoader(dataset, batch_size, shuffle=shuffle,
                      collate_fn=_pad_data_and_label(device))
