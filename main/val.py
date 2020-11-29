import argparse
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

from g2p.config import Config
from g2p.data import a2a_dataset, get_loader
from g2p.evaluation import levenshtein_distance, mean_score
from tools.log import get_simple_logger

if __name__ == '__main__':
    logger = get_simple_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default',
                        help="YAML file name under configs/")
    args = parser.parse_args()

    cfg = Config(args.config, train=False)

    device = cfg.DEVICE
    model = cfg.MODEL

    dataset = a2a_dataset
    _, val_ds = train_test_split(dataset, random_state=cfg.SEED)
    val_loader = get_loader(val_ds, cfg.BATCH_SIZE, device, shuffle=False)

    val_loss = 0
    val_dist = 0
    with torch.no_grad(), tqdm(val_loader) as pbar:
        for data, label in pbar:
            model.train()
            loss_item = model(data, label).item()
            val_loss += loss_item
            info = {'loss': loss_item}

            model.eval()
            pred = model(data, search_algo=cfg.SEARCH)
            dist = mean_score(levenshtein_distance, pred, label.seq)
            val_dist += dist
            info['dist'] = dist

            pbar.set_postfix({'loss': loss_item, 'dist': dist})

    val_loss /= len(val_loader)
    val_dist /= len(val_loader)

    logger.info(' Loss    Dist   \n'
               f'{val_loss:.4f}  {val_dist:.4f}')
