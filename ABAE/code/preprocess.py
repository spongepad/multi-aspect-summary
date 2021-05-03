import codecs
import json
import kss
import argparse

from tqdm import tqdm
from konlpy.tag import Okt

delete_tag = ['Josa', 'Suffix', 'KoreanParticle', 'Determiner', 'Punctuation', 'Foreign', 'Alpha', 'Exclamation', \
              'Hashtag', 'ScreenName', 'Email', 'URL', 'Number']
okt = Okt()

def parseSentence(line):
    tokens_pos = okt.pos(line, norm=True, stem=True)
    tokens_pos = [list(item) for item in tokens_pos]

    tokens_deltag = []
    for token in tokens_pos:
      if token[1] in delete_tag:
        continue
      tokens_deltag.append(token[0])

    return tokens_deltag


def preprocess_train(domain):
    f = codecs.open('/content/drive/MyDrive/Attention-Based-Aspect-Extraction-master/datasets/' + domain + '/train.txt', 'r', 'utf-8')
    out = codecs.open('/content/drive/MyDrive/Attention-Based-Aspect-Extraction-master/preprocessed_data/' + domain + '/train.txt', 'w', 'utf-8')

    for line in f:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            #sen_sp = kss.split_sentences(' '.join(tokens) + '\n')
            out.write(' '.join(tokens) + '\n')



def preprocess_test(domain):
    # For restaurant domain, only keep sentences with single 
    # aspect label that in {Food, Staff, Ambience}

    f1 = codecs.open('/content/drive/MyDrive/Attention-Based-Aspect-Extraction-master/datasets/' + domain + '/test.txt', 'r', 'utf-8')
    #f2 = codecs.open('/content/drive/MyDrive/Attention-Based-Aspect-Extraction-master/datasets/' + domain + '/test_label.txt', 'r', 'utf-8')
    out1 = codecs.open('/content/drive/MyDrive/Attention-Based-Aspect-Extraction-master/preprocessed_data/' + domain + '/test.txt', 'w', 'utf-8')
    #out2 = codecs.open('/content/drive/MyDrive/Attention-Based-Aspect-Extraction-master/preprocessed_data/' + domain + '/test_label.txt', 'w', 'utf-8')

    for text in f1:
        tokens = parseSentence(text)
        if len(tokens) > 0:
            #sen_sp = kss.split_sentences(' '.join(tokens) + '\n')
            out1.write(' '.join(tokens) + '\n')
    """
    for text, label in zip(f1, f2):
        label = label.strip()
        if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
            continue
        tokens = parseSentence(text)
        if len(tokens) > 0:
            sen_sp = kss.split_sentences(' '.join(tokens) + '\n')
            out1.write(', '.join(sen_sp) + '\n')
            out2.write(label + '\n')
    """


def preprocess(domain):
    print('\t' + domain + ' train set ...')
    preprocess_train(domain)
    print('\t' + domain + ' test set ...')
    preprocess_test(domain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant',
                        help="domain of the corpus")
    args = parser.parse_args()

    preprocess(args.domain)
