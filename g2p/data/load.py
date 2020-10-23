import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def _pad_data_and_label(batch):
    data, label = zip(*batch)
    return (pad_sequence(data), torch.LongTensor([len(x) for x in data])), \
           (pad_sequence(label), torch.LongTensor([len(y) for y in label]))

def get_loader(dataset, batch_size, use_cuda):
    kwargs = {'dataset': dataset, 'batch_size': batch_size,
              'collate_fn': _pad_data_and_label}
    kwargs.update({'num_workers': 1, 'pin_memory': True} if use_cuda else {})
    return DataLoader(**kwargs)
