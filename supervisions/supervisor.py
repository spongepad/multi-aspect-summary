import numpy as np
import pandas as pd
import kss
from sklearn.feature_extraction.text import TfidfVectorizer
from util import make_df, tfidf_tokenizer

from konlpy.tag import Okt
okt = Okt()

class Supervisor:
    def __init__(self):
        self._ae = make_df()

        self._tfidf_vectorizer = None
        self._document_vocab = None
        self._analyzer = None

    def get_aspects(self, text):
        in_text_aspect = []
        for aspect in self._ae.index.values:
          for relation in self._ae[aspect].split() :
            if relation in text:
              in_text_aspect.append(aspect)
            
        in_text_aspect = list(set(in_text_aspect))


        return in_text_aspect

    def guess_summary(self, aspect, document, rel_words):
        neighbors = [{
            'entity': aspect,
            'reasoning': f'aspect [[{aspect}]] is in the text',
            'relation_weight': float('inf')}]

        for rel_word in rel_words:
            neighbors.append({
              'entity': rel_word,
              'reasoning': f'Related word [[{rel_word}]] is in the text',
              'relation_weight': float('inf')})

        picked_sents, reasonings = [], []
        for sent in kss.split_sentences(document):
            sent_morphs = okt.morphs(sent)
            for neighbor in neighbors:
                if neighbor['entity'] in sent_morphs:
                    if sent not in picked_sents:
                        picked_sents.append(sent)
                    if neighbor['reasoning'] not in reasonings:
                        reasonings.append(neighbor['reasoning'])

        if len(picked_sents) > 0:
            return {
                'aspect_summary': ' '.join(picked_sents),
                'reasonings': reasonings
            }
        else:
            return None


    def get_rel_words(self, aspect, document, n_limit):
        document_words = okt.morphs(document)

        selected_words = []
        rel_words = self._ae[aspect].split()

        for word in document_words:
            word_stem = okt.morphs(word, norm=True, stem=True)
            if word_stem[0] in rel_words:
                selected_words.append(word)

            if len(selected_words) == n_limit:
                break

        selected_words = list(set(selected_words))
        return selected_words
    '''
    def build_tfidf_vectorizer(self, documents):
        print('fitting tfidf ...', end=' ')
        self._tfidf_vectorizer = TfidfVectorizer(tokenizer=tfidf_tokenizer)
        self._tfidf_vectorizer.fit(documents)
        self._document_vocab = self._tfidf_vectorizer.get_feature_names()
        self._analyzer = self._tfidf_vectorizer.build_analyzer()
        print('done')

    def get_doc_words(self, document):
        vec = self._tfidf_vectorizer.transform([document])
        word_id_score_list = [[self._document_vocab[word_id], word_id, score]
                              for word_id, score in zip(vec.indices, vec.data)]

        word_id_score_list.sort(key=lambda t: t[-1], reverse=True)

        return [word for word, _, _ in word_id_score_list]'''
