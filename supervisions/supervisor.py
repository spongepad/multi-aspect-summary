import numpy as np
import kss
from utils.util import make_df

from konlpy.tag import Okt
okt = Okt()

class Supervisor:
    def __init__(self):
        self._ae = make_df()


    def get_aspects(self, text):
        in_text_aspect = []
        
        for aspect in self._ae.index.values:
            if aspect in text:
                if aspect not in in_text_aspect:
                    in_text_aspect.append(aspect)

            for relation in self._ae[aspect].split() :
                if relation in text:
                    if aspect not in in_text_aspect:
                        in_text_aspect.append(aspect)
            

        return in_text_aspect

    def guess_summary(self, aspect, document, rel_words):
        neighbors = [{'entity': aspect}]

        for rel_word in rel_words:
            neighbors.append({'entity': rel_word})

        picked_sents = []
        for sent in kss.split_sentences(document):
            for neighbor in neighbors:
                if neighbor['entity'] in sent:
                    if sent not in picked_sents:
                        picked_sents.append(sent)

        if len(picked_sents) > 0:
            return {'aspect_summary': ' '.join(picked_sents)}
        else:
            return None


    def get_rel_words(self, aspect, document, n_limit):
        document_words = okt.morphs(document)

        selected_words = []
        rel_words = self._ae[aspect].split()

        if aspect not in rel_words:
            rel_words.append(aspect)

        for word in document_words:
            word_stem = okt.morphs(word, norm=True, stem=True)
            if word_stem[0] in rel_words:
                if word not in selected_words:
                    selected_words.append(word)

            if len(selected_words) == n_limit:
                break

        return selected_words