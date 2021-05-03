# -*- coding: utf-8 -*-
import os
import fire
import json
from tqdm import tqdm
import pickle

from supervisions.supervisor import Supervisor


N_REL_WORDS = 20


def load_data(split, n_examples):
    src_file = open(f'data/earphone/{split}.source', encoding='utf-8')
    #tgt_file = open(f'data/cnn_dm/{split}.target', encoding='utf-8')

    documents = []
    for src in src_file.readlines()[:n_examples]:
        documents.append(src.strip())

    return documents


def main(split):
    train_documents = load_data(split='train', n_examples=None)
    supervisor = Supervisor()
    #supervisor.build_tfidf_vectorizer(documents=train_documents)
    pickle.dump(supervisor, open('supervisions/supervisor.pickle', 'wb'))
    print('supervisor initialized.')

    documents = load_data(split=split, n_examples=None)

    for l in range(0, len(documents), 10000):
        r = min(len(documents), l + 10000)

        if os.path.exists(f'{split}_{l}-{r}.json'):
            continue

        dataset = []
        json.dump(dataset, open(f'data/earphone/{split}_{l}-{r}.json', 'w'), indent=4, ensure_ascii=False)

        prog_bar = tqdm(documents[l:r], total=len(documents[l:r]))
        for doc_id, (document) in enumerate(prog_bar):
            in_text_aspects = supervisor.get_aspects(document)

            dataset.append({
                'doc_id': doc_id,
                'document': document,
                'aspect_summaries': []
            })

            prog_bar.set_postfix_str(
                f'in_text_ents: {len(in_text_aspects)}')
            for aspect in in_text_aspects:
                rel_words = supervisor.get_rel_words(
                    aspect=aspect, document=document, n_limit=N_REL_WORDS)

                guessed_summary = supervisor.guess_summary(
                    aspect=aspect, document=document, rel_words=rel_words)

                if guessed_summary is not None:
                    dataset[-1]['aspect_summaries'].append({
                        'aspect': aspect,
                        'summary': guessed_summary['aspect_summary'],
                        'reasonings': guessed_summary['reasonings'],
                        'rel_words': rel_words
                    })

            json.dump(dataset, open(f'data/earphone/{split}_{l}-{r}.json', 'w'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
    fire.Fire(main)
