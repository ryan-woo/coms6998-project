import os
import json
import string
from torch import Tensor

from tokenizers.models import WordLevel, BPE
from tokenizers import ByteLevelBPETokenizer

# from tokenizers import WordLevelTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import Tokenizer, PreTokenizedString, NormalizedString
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import PreTokenizer
from transformers import GPT2Model, GPT2Tokenizer

from tokenizers import ByteLevelBPETokenizer


class CharPreTokenizer:

    SPACE_CHAR = "[SPACE]"

    @staticmethod
    def split_by_char(i: int, normalized_string: NormalizedString):
        # Not totally sure why parameter i is necessary?
        # It exists in the example https://github.com/huggingface/tokenizers/blob/master/bindings/python/examples/custom_components.py
        split = []
        for c in str(normalized_string):
            if c == " ":
                c = CharPreTokenizer.SPACE_CHAR
            split.append(NormalizedString(c))
        return split

    def pre_tokenize_str(self, sequence: str):

        split = list(sequence)  # Converts "123" -> ["1", "2", "3"]

        output = []
        for i, c in enumerate(split):
            if c == " ":
                c = CharPreTokenizer.SPACE_CHAR
            output.append((c, (i, i+1)))
        return output

    def pre_tokenize(self, pretok: PreTokenizedString):

        pretok.split(self.split_by_char)

def create_char_tokenizer():

    ascii_table = {str(chr(i)): i for i in range(128)}
    # ascii_table = {c: ord(c) for c in string.ascii_letters + string.digits + string.punctuation}
    vocab = {"[SPACE]": 128, "<s>": 129, "<pad>": 130, "</s>": 131, "<unk>": 132, "<mask>": 134}
    vocab.update(ascii_table)
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = PreTokenizer.custom(CharPreTokenizer())

    return tokenizer


def main():

    # tokenizer = create_char_tokenizer()
    # os.mkdir("chartok")
    # tokenizer.model.save("chartok")
    # open("chartok/merges.txt", "w").close()
    model = GPT2Model.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("chartok")

    v = gpt2_tokenizer.get_vocab()
    print(v)

    hi = gpt2_tokenizer(["aaa¥bbb"], return_tensors="pt")
    print(hi["input_ids"])
    embed = model.wte(hi["input_ids"])
    print(embed)

if __name__ == "__main__":
    main()