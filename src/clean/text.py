from typing import Dict, List, Optional, Set, Tuple

import torch
from torchtext.data import get_tokenizer
from torchtext.experimental import transforms
from torchtext.vocab import Vocab

from clean import vocab as V

local_vocab = V.load_vocab()


def clean(txt: str, vocab: Vocab=local_vocab) -> List[int]:
    txt = "<start> {} <end>".format(txt.strip())
    return [vocab.stoi[word] for word in V.clean_sentence(txt)]

def clean_batch(text: List[str], vocab: Vocab=local_vocab) -> torch.tensor:
    text = ["<start> {} <end>".format(txt.strip()) for txt in text]
    sentences = [V.clean_sentence(txt) for txt in text]
    pad_len = max([len(x) for x in sentences])
    sentences = [pad(x, pad_len) for x in sentences]
    return torch.tensor([[vocab.stoi[word] for word in sentence] for sentence in sentences])

def pad(sentence: List[str], length: int, sym: str='<pad>'):
    while len(sentence) < length:
        sentence = sentence + [sym]
    return sentence

if __name__ == "__main__":
    print(clean("The quick brown fox jumps over the lazy dog!"))
