import os
import fire
import pickle

from data_utils.datasets import  SummaryDataset
from models.bart import BART

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.core.saving import save_hparams_to_yaml


from loguru import logger


BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 10
LR = 3e-5
NUM_WORKERS = 6
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
WARMUP_PROPORTION = 0.05
MAX_SRC_LENGTH = 512
MAX_TGT_LENGTH = 140
LABEL_SMOOTH = 0.1
PRECISION = 16
N_WIKI_WORDS = 20
# MASK_RATIO = 1.0
MODEL_INIT = 'kobart'

class OnCheckpointHparams(Callback):
    def on_save_checkpoint(self, trainer, pl_module):
        # only do this 1 time
        if trainer.current_epoch == 0:
            file_path = f"{trainer.logger.log_dir}/hparams.yaml"
            print(f"Saving hparams to file_path: {file_path}")
            save_hparams_to_yaml(config_yaml=file_path, hparams=pl_module.hparams)


def get_dataset(dataset_name, train_docs, related_word_mask, rel_flag = True, MASK_RATIO=1.0):
    return {split: SummaryDataset(
        split=split,
        domain=dataset_name, 
        max_src_length=MAX_SRC_LENGTH, 
        max_tgt_length=MAX_TGT_LENGTH,
        rel_flag = rel_flag,
        mask_ratio=MASK_RATIO,
        n_docs=train_docs if split == 'train_no_aug' else None,
        related_word_mask=related_word_mask)
        for split in ['train_no_aug', 'val_no_aug']}


def main(dataset_name='weaksup', n_epochs=1, train_docs=100,
         pretrained_ckpt=None, related_word_mask=True,
         dir_path='0', rel_flag=True , mask_ratio=1.0):
    dataset = get_dataset(dataset_name=dataset_name, train_docs=train_docs, related_word_mask=related_word_mask, rel_flag=rel_flag ,MASK_RATIO=mask_ratio)

    dataloaders = {
        split: DataLoader(
            dataset=dataset[split],
            batch_size=BATCH_SIZE,
            shuffle=(split == 'train_no_aug'),
            num_workers=NUM_WORKERS)
        for split in ['train_no_aug', 'val_no_aug']}

    if pretrained_ckpt is not None:
        log_dir = f'logs/{dataset_name}_plus/docs{dir_path}/'
    else:
        log_dir = f'logs/{dataset_name}/docs{dir_path}/'

    if os.path.exists(log_dir):
        print(f'log_dir \"{log_dir}\" exists. training skipped.')
        return
    os.makedirs(log_dir)

    logger.add(f'{log_dir}/log.txt')
    logger.info(f'pretrained checkpoint: {pretrained_ckpt}')
    if pretrained_ckpt is None:
        bart = BART(text_logger=logger)
    else:
        bart = BART.load_from_checkpoint(
            checkpoint_path=pretrained_ckpt, text_logger=logger)

    train_steps = n_epochs * (
            len(dataloaders['train_no_aug']) // ACCUMULATE_GRAD_BATCHES + 1)
    warmup_steps = int(train_steps * WARMUP_PROPORTION)
    
    bart.set_hparams(
        batch_size=BATCH_SIZE,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON,
        label_smooth=LABEL_SMOOTH)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{log_dir}/best_model',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True)

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, OnCheckpointHparams()],
        max_epochs=n_epochs,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gpus=1,
        precision=PRECISION)

    trainer.fit(
        model=bart,
        train_dataloader=dataloaders['train_no_aug'],
        val_dataloaders=dataloaders['val_no_aug'])

    logger.info('training finished.')


if __name__ == '__main__':
    fire.Fire(main)