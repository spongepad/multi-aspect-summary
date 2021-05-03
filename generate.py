import os
import fire
import pickle
import cleantext
import torch
from tqdm import trange

from data_utils.datasets import SummaryDataset
from models.bart import BART
from kobart import get_kobart_tokenizer

from transformers import BartForConditionalGeneration

BATCH_SIZE = 1
MAX_LEN = 140
LEN_PENALTY = 2.
BEAM_SIZE = 4
NO_REPEAT_NGRAM_SIZE = 2

def main(log_path, wiki_sup=True):
    supervisor = pickle.load(open('supervisions/supervisor.pickle', 'rb')) \
        if wiki_sup else None
    dataset = SummaryDataset(split='test',
        domain='earphone', 
        max_src_length=512, 
        max_tgt_length=MAX_LEN)
    test_examples = [example for example in dataset]

    tokenizer = get_kobart_tokenizer()
    
    bart = BartForConditionalGeneration.from_pretrained(f'{log_path}')

    src_file = open(f'{log_path}/test.source', 'w')
    gold_file = open(f'{log_path}/test.gold', 'w')
    hypo_file = open(f'{log_path}/test.hypo', 'w', encoding='utf-8')

    for i in trange(0, len(test_examples[:10]), BATCH_SIZE, desc=f'Generating'):
        batch_examples = test_examples[i:i+BATCH_SIZE]
        
        for example in batch_examples:
          output = bart.generate(
            example['input_ids'].unsqueeze(0),
            max_length=MAX_LEN,
            num_beams=BEAM_SIZE,
            no_repeat_ngram_size = NO_REPEAT_NGRAM_SIZE,
            length_penalty = LEN_PENALTY)

          output = tokenizer.decode(output[0], skip_special_tokens=True)

          print(example['src'].replace('\n\n', ' ||| '), file=src_file)
          print(example['tgt'], file=gold_file)
          print(output, file=hypo_file)
          print('\n',output)

if __name__ == '__main__':
    fire.Fire(main)