import os
import re
import json
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BartTokenizer

from kobart import get_kobart_tokenizer

class SummaryDataset(Dataset):
    def __init__(self, split, domain, max_src_length, max_tgt_length, ignore_index=-100,
    mask_ratio=0, n_docs=None,weak_sup = True):

        self.tokenizer = get_kobart_tokenizer()
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.ignore_index = ignore_index
        self.bos_token = '<s>'
        self.eos_token = '</s>'

        self.mask_ratio = mask_ratio

        self.masking = True if mask_ratio > 0 else False
        self.weak_sup = weak_sup
        
        print('@@@@@@@@@@@@@@@@@@@@@split : ', split)

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

    def noise_sentence(self, document, rel_words, mask_ratio, replacement_token = "<mask>"):

        num_words = int(len(rel_words) * mask_ratio)

        if num_words == 0 and len(rel_words) > 0: num_words+=1

        sample_rel_word = random.sample(rel_words, num_words)
        for rel in sample_rel_word:
          document = re.sub(pattern = rel, repl="<mask>" ,string=document)
        
        document = re.sub(r'<mask> <mask>', "<mask>", document)
        document = re.sub(r'<mask> <mask>', "<mask>", document)
        return document

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        example = self._examples[item]
        
        if self.weak_sup:
            if self.masking :
              src = '{bos}{aspect} : {rel_words}\n\n{doc}{eos}'.format(
                aspect=example['aspect'],
                rel_words=' '.join(example['rel_words']),
                doc=self.noise_sentence(example['document'], example['rel_words'], self.mask_ratio),
                bos=self.bos_token,
                eos=self.eos_token)
            else :
              src = '{bos}{aspect} : {rel_words}\n\n{doc}{eos}'.format(
                aspect=example['aspect'],
                rel_words=' '.join(example['rel_words']),
                doc=example['document'],
                bos=self.bos_token,
                eos=self.eos_token)
                
        else:
            if self.masking :
              src = '{bos}{aspect}\n\n{doc}{eos}'.format(
                doc=self.noise_sentence(example['document'], example['rel_words'], self.mask_ratio),
                bos=self.bos_token,
                eos=self.eos_token)
            else :
              src = '{bos}{aspect}\n\n{doc}{eos}'.format(
                doc=example['document'],
                bos=self.bos_token,
                eos=self.eos_token)
                
        # print('src : ', src) # 출력 잘되는지 확인하기

        tgt = '{bos}{summary}{eos}'.format(
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
