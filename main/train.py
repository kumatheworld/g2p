import argparse
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.tensorboard import SummaryWriter

from g2p.config import Config
from g2p.data import a2a_dataset, get_loader, DoubleBets
from g2p.evaluation import levenshtein_distance, mean_score
from tools.nop import Nop
from tools.log import get_simple_logger

if __name__ == '__main__':
    logger = get_simple_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default',
                        help="YAML file name under configs/")
    args = parser.parse_args()

    logger.info('Loading config...')
    cfg = Config(args.config, train=True)

    device = cfg.DEVICE
    model = cfg.MODEL

    logger.info('Loading data...')
    dataset = a2a_dataset
    if cfg.SANITY_CHECK.EN:
        dataset = random.sample(dataset, cfg.SANITY_CHECK.NUM_DATA)

    if cfg.VALIDATE:
        train_ds, test_ds = train_test_split(dataset, random_state=cfg.SEED)
        train_loader = get_loader(train_ds, cfg.BATCH_SIZE, device)
        val_loader = get_loader(test_ds, cfg.BATCH_SIZE, device)
    else:
        train_loader = get_loader(dataset, cfg.BATCH_SIZE, device)
        val_loader = None

    optimizer = cfg.OPTIMIZER
    lr_scheduler = cfg.LR.SCHEDULER
    writer = Nop() if cfg.SANITY_CHECK.EN else \
             SummaryWriter(comment=f'-{cfg.name}')
    writer.add_text('config', str(cfg))

    n_iter = 1
    best_dist = float('inf')
    for epoch in range(1, cfg.EPOCHS + 1):
        # train
        train_loss = 0
        train_dist = 0
        for data, label in train_loader:
            model.train()
            optimizer.zero_grad()
            loss = model(data, label)
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            writer.add_scalar('loss/train', loss_item, n_iter)
            train_loss += loss_item

            model.eval()
            with torch.no_grad():
                pred = model(data, search_algo=cfg.SEARCH)
                label_seq = label[0]
                dist = mean_score(levenshtein_distance, pred, label_seq)
            writer.add_scalar('dist/train', dist, n_iter)
            train_dist += dist

            logger.debug(f'Train:{n_iter:8d}  {loss_item:.4f}  {dist:.4f}')

            n_iter += 1

        train_loss /= len(train_loader)
        train_dist /= len(train_loader)

        # validate
        val_loss = 0
        val_dist = 0
        if cfg.VALIDATE:
            with torch.no_grad():
                for data, label in val_loader:
                    model.train()
                    loss = model(data, label)
                    val_loss += loss

                    model.eval()
                    pred = model(data, search_algo=cfg.SEARCH)
                    label_seq = label[0]
                    dist = mean_score(levenshtein_distance, pred, label_seq)
                    val_dist += dist

            val_loss /= len(val_loader)
            val_dist /= len(val_loader)
            writer.add_scalars('loss/train & val',
                            {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('dist/train & val',
                            {'train': train_dist, 'val': val_dist}, epoch)

            logger.debug(f'  Val:{epoch:8d}  {val_loss:.4f}  {val_dist:.4f}')

        # visualize embeddings
        writer.add_embedding(model.enc_emb.weight, DoubleBets.alphabet.i2t,
                             global_step=epoch, tag='Alphabet')
        writer.add_embedding(model.dec_emb.weight,
                             DoubleBets.arpabet.special_tokens + \
                             list(DoubleBets.arpabet2ipa.values()),
                             global_step=epoch, tag='IPA')

        # save model
        if best_dist >= val_dist and not cfg.SANITY_CHECK.EN:
            best_dist = val_dist
            checkpoint = {
                'config': cfg.dictionary,
                'epoch': epoch,
                'model': model.state_dict(),
            }
            torch.save(checkpoint, cfg.CKPT_PATH)

        lr_scheduler.step()

    writer.close()
