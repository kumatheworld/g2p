from types import SimpleNamespace
import yaml
import os
import random
import numpy as np
import torch
import torch.optim as optim

from g2p.data import DoubleBets
from g2p import models
from g2p.processing import search

def _fix_seed(seed):
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _attributify(obj, dikt):
    for k, v in dikt.items():
        if isinstance(v, dict):
            setattr(obj, k, SimpleNamespace())
            _attributify(getattr(obj, k), v)
        else:
            setattr(obj, k, v)

class Config:
    def __init__(self, config_name, train):
        self.name = config_name
        config_dir = 'configs'
        config_path = os.path.join(config_dir, config_name + '.yaml')

        with open(config_path) as f:
            self.yaml_str = f.read()
        cfg = yaml.safe_load(self.yaml_str)

        _attributify(self, cfg)

        if cfg['SEED'] is not None:
            _fix_seed(cfg['SEED'])

        use_cuda = torch.cuda.is_available() and cfg['USE_CUDA']
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.USE_CUDA = use_cuda
        self.DEVICE = device

        model = getattr(models, cfg['MODEL']['NAME'])(
            src_size=len(DoubleBets.alphabet),
            tgt_size=len(DoubleBets.arpabet),
             **cfg['MODEL']['KWARGS'],
        )
        model.to(device)
        self.MODEL = model

        self.SEARCH = getattr(search, cfg['SEARCH']['ALGO'])(
            **cfg['SEARCH']['KWARGS']
        )

        if train:
            self.OPTIMIZER = getattr(optim, cfg['OPTIMIZER'])(
                model.parameters(), lr=cfg['LR']
            )
        else:
            ckpt = torch.load(self.CKPT_PATH, map_location=device)
            model.load_state_dict(ckpt['model'])

    def get_markdown_string(self):
        return f'<pre>{self.str}</pre>'
