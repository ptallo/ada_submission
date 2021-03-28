import json
from collections import Counter
from itertools import chain
from typing import Iterator, List

import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from torchtext.vocab import Vocab

nltk.download('punkt')
nltk.download('stopwords')

stop_words = stopwords.words('english')
porter = PorterStemmer()


def get_annotations_from_file(path) -> Iterator[str]:
    with open(path, 'r') as f:
        data = json.loads(f.read())
        for annotation_json in data['annotations']:
            yield annotation_json['caption']


def clean_sentence(txt: str) -> List[str]:
    return ["<start>"] + [porter.stem(word.lower()) for word in tokenize.word_tokenize(txt) if should_keep_word(word)] + ["<end>"]


def should_keep_word(text: str) -> bool:
    return text.isalpha() and text not in stop_words


def save_vocab(file_path: str, vocab: Vocab):
    with open(file_path, 'w') as f:
        f.writelines(["{} {}\n".format(token, index) for (token, index) in vocab.stoi.items()])


def load_vocab(file_path: str='vocab.txt') -> Vocab:
    stoi = {}
    itos = {}
    with open(file_path, 'r') as f:
        lines = [x.split() for x in f.readlines()]
        for (word, idx) in lines:
            stoi[word] = int(idx)
            itos[int(idx)] = word

        vocab = Vocab(Counter())
        vocab.stoi = stoi
        vocab.itos = itos
        return vocab


if __name__ == "__main__":
    joint_iterator = chain(
        get_annotations_from_file('coco/annotations/captions_train2017.json'),
        get_annotations_from_file('coco/annotations/captions_val2017.json'),
    )

    counter = Counter()
    for sentence in joint_iterator:
        counter.update(clean_sentence(sentence))
    vocab = Vocab(counter, min_freq=1)
    save_vocab('vocab.txt', vocab)
    print(len(vocab))
