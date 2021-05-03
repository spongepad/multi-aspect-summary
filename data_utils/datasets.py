import os
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BartTokenizer

from kobart import get_kobart_tokenizer

class SummaryDataset(Dataset):
    def __init__(self, split, domain, max_src_length, max_tgt_length, ignore_index=-100, n_docs=None):

        self.tokenizer = get_kobart_tokenizer()
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.ignore_index = ignore_index
        self.bos_token = '<s>'
        self.eos_token = '</s>'

        data_path = f'data/{domain}/{split}.json'

        docs = json.load(open(data_path))
        docs = docs[:n_docs]

        self._examples = []
        for doc in docs:
            for asp_sum in doc['aspect_summaries']:
                self._examples.append({
                    'aspect': asp_sum['aspect'],
                    'rel_words': asp_sum['rel_words'],
                    'document': doc['document'],
                    'summary': asp_sum['summary']
                })

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        example = self._examples[item]

        src = '{bos}{aspect} : {rel_words}\n\n{doc}{eos}'.format(
          aspect=example['aspect'],
          rel_words=' '.join(example['rel_words']),
          doc=example['document'],
          bos=self.bos_token,
          eos=self.eos_token)

        tgt = '{bos}{aspect} : {summary}{eos}'.format(
          aspect=example['aspect'],
          summary=example['summary'],
          bos=self.bos_token,
          eos=self.eos_token)
        
        input_ids = self.tokenizer(src, 
                max_length=self.max_src_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt')

        labels = self.tokenizer(tgt, 
                max_length=self.max_tgt_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt')

        return {
          'input_ids': input_ids['input_ids'].squeeze(),
          'attention_mask': input_ids['attention_mask'].squeeze(),
          'labels': labels['input_ids'].squeeze(),
          'src': src,
          'tgt': tgt
          } 
