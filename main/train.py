import argparse
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
    if cfg.NUM_DATA >= 0:
        dataset = random.sample(dataset, cfg.NUM_DATA)

    if cfg.VALIDATE:
        train_ds, val_ds = train_test_split(dataset, random_state=cfg.SEED)
        train_loader = get_loader(train_ds, cfg.BATCH_SIZE, device)
        val_loader = get_loader(val_ds, cfg.BATCH_SIZE, device, shuffle=False)
    else:
        train_loader = get_loader(dataset, cfg.BATCH_SIZE, device)
        val_loader = None

    optimizer = cfg.OPTIMIZER
    lr_scheduler = cfg.LR.SCHEDULER
    writer = Nop() if cfg.SANITY_CHECK else \
             SummaryWriter(comment=f'-{cfg.name}')
    writer.add_text('config', str(cfg))

    logger.info('Start training!')
    n_iter = 1
    best_loss = float('inf')
    best_dist = float('inf')
    for epoch in range(1, cfg.EPOCHS + 1):
        logger.info(f'Epoch {epoch} / {cfg.EPOCHS}')
        model.train()
        train_loss = 0
        train_dist = 0
        with tqdm(train_loader, desc='[Train]') as pbar:
            for data, label in pbar:
                optimizer.zero_grad()
                loss = model(data, label)
                loss.backward()
                optimizer.step()

                loss_item = loss.item()
                writer.add_scalar('loss/train', loss_item, n_iter)
                train_loss += loss_item
                info = {'loss': loss_item}

                if cfg.EVAL_TRAIN:
                    model.eval()
                    with torch.no_grad():
                        pred = model(data, search_algo=cfg.SEARCH)
                        dist = mean_score(levenshtein_distance,
                                          pred, label.seq)
                    writer.add_scalar('dist/train', dist, n_iter)
                    train_dist += dist
                    info['dist'] = dist
                    model.train()

                pbar.set_postfix(info)
                n_iter += 1

        train_loss /= len(train_loader)
        train_dist /= len(train_loader)

        # validate
        model.eval()
        val_loss = 0
        val_dist = 0
        if cfg.VALIDATE:
            with torch.no_grad(), \
                 tqdm(val_loader, desc='  [Val]') as pbar:
                for data, label in pbar:
                    loss_item = model(data, label).item()
                    val_loss += loss_item
                    info = {'loss': loss_item}

                    if cfg.EVAL_VAL:
                        pred = model(data, search_algo=cfg.SEARCH)
                        dist = mean_score(levenshtein_distance,
                                          pred, label.seq)
                        val_dist += dist
                        info['dist'] = dist

                    pbar.set_postfix(info)

            val_loss /= len(val_loader)
            val_dist /= len(val_loader)
            writer.add_scalars('loss/train & val',
                               {'train': train_loss, 'val': val_loss}, epoch)
            info = {}
            if cfg.EVAL_TRAIN:
                info['train'] = train_dist
            if cfg.EVAL_VAL:
                info['val'] = val_dist
            writer.add_scalars('dist/train & val', info, epoch)

        logger.info('Summary:\n'
                    '         Loss    Dist   \n'
                   f' Train  {train_loss:.4f}  ' +
                    (f'{train_dist:.4f}' if cfg.EVAL_TRAIN else "------"))
        if cfg.VALIDATE:
            logger.info(f'   Val  {val_loss:.4f}  ' +
                        (f'{val_dist:.4f}' if cfg.EVAL_VAL else "------"))

        # visualize embeddings
        writer.add_embedding(model.enc_emb.weight, DoubleBets.alphabet.i2t,
                             global_step=epoch, tag='Alphabet')
        writer.add_embedding(model.dec_emb.weight,
                             DoubleBets.arpabet.special_tokens + \
                             list(DoubleBets.arpabet2ipa.values()),
                             global_step=epoch, tag='IPA')

        # save checkpoint
        if (best_dist >= val_dist if cfg.EVAL_VAL else best_loss >= val_loss) \
            and not cfg.SANITY_CHECK:
            if cfg.VALIDATE:
                logger.info('Best accuracy achieved!')
            logger.info('Saving model...')
            if cfg.EVAL_VAL:
                best_dist = val_dist
            else:
                best_loss = val_loss
            checkpoint = {
                'config': cfg.dictionary,
                'epoch': epoch,
                'model': model.state_dict(),
            }
            torch.save(checkpoint, cfg.CKPT_PATH)

        lr_scheduler.step()
        logger.info('')

    writer.close()
    logger.info('Training done!')
