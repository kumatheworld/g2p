import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def _pad_data_and_label(device):
    def _pad_and_get_len(seqs):
        return pad_sequence(seqs).to(device), \
               torch.LongTensor([len(seq) for seq in seqs])

    def _pad_data_and_label_(batch):
        data, label = zip(*batch)
        return _pad_and_get_len(data), _pad_and_get_len(label)

    return _pad_data_and_label_

def get_loader(dataset, batch_size, device):
    return DataLoader(dataset, batch_size, shuffle=True,
                      collate_fn=_pad_data_and_label(device))
