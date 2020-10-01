from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader

def _pack_data_and_label(batch):
    data, label = zip(*batch)
    return pack_sequence(data, enforce_sorted=False), \
           pack_sequence(label, enforce_sorted=False)

def get_loader(dataset, batch_size, use_cuda):
    kwargs = {'dataset': dataset, 'batch_size': batch_size,
              'collate_fn': _pack_data_and_label}
    kwargs.update({'num_workers': 1, 'pin_memory': True} if use_cuda else {})
    return DataLoader(**kwargs)
