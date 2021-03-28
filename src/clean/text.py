from typing import Dict, List, Optional, Set, Tuple

import torchtext
from torchtext.data import get_tokenizer
from torchtext.experimental import transforms

import vocab as V

vocab = V.load_vocab()


def clean(txt: str) -> List[int]:
    txt = "<start> {} <end>".format(txt.trim())
    return [vocab.stoi[word] for word in V.clean_sentence(txt)]


if __name__ == "__main__":
    print(clean("The quick brown fox jumps over the lazy dog!"))
