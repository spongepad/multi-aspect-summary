import gensim
import codecs


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main(domain):
    source = '/content/drive/MyDrive/Attention-Based-Aspect-Extraction-master/preprocessed_data/%s/train.txt' % (domain)
    model_file = '/content/drive/MyDrive/Attention-Based-Aspect-Extraction-master/preprocessed_data/%s/w2v_embedding' % (domain)
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=4, min_count=10, workers=4, sg=0, negative=5)
    model.save(model_file)


print('Pre-training word embeddings ...')
main('earphone')
# main('beer')
# main('laptops')
