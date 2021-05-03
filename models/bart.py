import os

import torch
from torch import nn

import numpy as np
import pytorch_lightning as pl

from transformers import BartTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration, BartModel

from kobart import get_pytorch_kobart_model, get_kobart_tokenizer


class BART(pl.LightningModule):

    def __init__(self, hparam=None, text_logger=None):
        super(BART, self).__init__()

        self._model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
        self._model.train()
        self.tokenizer = get_kobart_tokenizer()

        self._hparams = hparam

        self._text_logger = text_logger

    def forward(self, inputs):
        return self._model(input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'],
                          labels=inputs['labels'])

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
          outs = self(batch)
          loss = outs.loss

        return loss.item()

    def validation_epoch_end(self, val_step_outputs):
        val_loss = sum(val_step_outputs) / len(val_step_outputs)

        self.log('val_loss', val_loss)

        lr = self.optimizers().state_dict()['param_groups'][0]['lr']
        self._text_logger.info(
            f'epoch {self.current_epoch}, lr = {lr}, val loss = {val_loss}')

    def set_hparams(self, **kwargs):
        self._hparams = kwargs

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": self._hparams['weight_decay']},
            {"params": [p for n, p in self._model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self._hparams['lr'], eps=self._hparams['adam_epsilon'])

        lr_scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self._hparams['warmup_steps'],
                num_training_steps=self._hparams['train_steps']),
            'monitor': 'loss', 'interval': 'step',
            'frequency': 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}