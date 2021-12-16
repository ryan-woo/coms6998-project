import os 
import re 
import pickle
import logging
import warnings
from pathlib import Path

import scipy
import epitran
from lazy_load import lazy

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

from candbproj import util
from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import PassageTokenizer
from candbproj.result import PereiraResult, PereiraResultSet, Args


logging.basicConfig()
log = logging.getLogger()

# IPA characters which appear in Pereira et al.

IPA_VOCAB = {
    "$": 1,
    "%": 2,
    ",": 3,
    "-": 4,
    ".": 5,
    "0": 6,
    "1": 7,
    "4": 8,
    "7": 9,
    "8": 10,
    "9": 11,
    ":": 12,
    "a": 13,
    "b": 14,
    "d": 15,
    "e": 16,
    "f": 17,
    "h": 18,
    "i": 19,
    "j": 20,
    "k": 21,
    "l": 22,
    "m": 23,
    "n": 24,
    "o": 25,
    "p": 26,
    "s": 27,
    "t": 28,
    "u": 29,
    "v": 30,
    "w": 31,
    "z": 32,
    "\u00e6": 33,
    "\u00f0": 34,
    "\u014b": 35,
    "\u0251": 36,
    "\u0254": 37,
    "\u0259": 38,
    "\u025b": 39,
    "\u0261": 40,
    "\u026a": 41,
    "\u0279": 42,
    "\u0283": 43,
    "\u028a": 44,
    "\u028c": 45,
    "\u0292": 46,
    "\u0329": 47,
    "\u0361": 48,
    "\u03b8": 49,
    "~": 50
}

class IpaTokenizerResult(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def char_to_token(self, index):
        return index

class IpaTokenizer:
    def __call__(self, sentences, add_special_tokens, return_tensors):
        assert add_special_tokens
        assert return_tensors == 'pt'

        input_ids = []
        attention_masks = []
        for sentence in sentences:
            input_ids.append([])
            for char in sentence:
                input_ids[-1].append(IPA_VOCAB[char])

            input_ids[-1] = torch.tensor(input_ids[-1])
            attention_masks.append(torch.ones(input_ids[-1].shape[0]))

        result = IpaTokenizerResult()
        result['input_ids'] = pad_sequence(
            input_ids, batch_first=True, padding_value=0)
        result['attention_mask'] = pad_sequence(
            attention_masks, batch_first=True, padding_value=0)
        return result

EPI = lazy(lambda: epitran.Epitran('eng-Latn'))
PREPROCESSING_DIR = Path(__file__).parent.resolve() / '../preprocessed'
TRANSLITERTIONS_CACHE_FILE = os.path.join(PREPROCESSING_DIR, 'ipa.pkl')
TRANSLITERATIONS = None

def ipa_preprocessor(sentence_id, sentence):
    global TRANSLITERATIONS

    if TRANSLITERATIONS is None:
        if os.path.exists(TRANSLITERTIONS_CACHE_FILE):
            with open(TRANSLITERTIONS_CACHE_FILE, 'rb') as f:
                TRANSLITERATIONS = pickle.load(f)
        else:
            TRANSLITERATIONS = {}

    if sentence_id not in TRANSLITERATIONS:
        TRANSLITERATIONS[sentence_id] = EPI.transliterate(sentence)
        with open(TRANSLITERTIONS_CACHE_FILE, 'wb') as f:
            pickle.dump(TRANSLITERATIONS, f)

    return re.sub("\s", "~", TRANSLITERATIONS[sentence_id])

def main():
    args = util.parse_args()

    gpt2_ipa_char_tokenizer_result = Path(__file__).parent.resolve() / "../../results/gpt2_ipa_char_tokenizer_result.pkl"
    if gpt2_ipa_char_tokenizer_result.exists():
        with open(gpt2_ipa_char_tokenizer_result, "rb") as f:
            result_set = pickle.load(f)
    else:
        results = []
        for seed in range(0, 10000, int(10000/args.n)):
            util.seeder(seed)

            model_config = GPT2Config.from_pretrained("gpt2")
            assert model_config.vocab_size > 0
            model_config.vocab_size = len(IPA_VOCAB) + 1
            model_config.output_hidden_states = True

            model_args = Args(
                args = ("gpt2",),
                kwargs = {
                    "vocab_size": model_config.vocab_size,
                    "output_hidden_states": True
                }
            )
            model = GPT2Model(model_config)
            model = model.eval()

            tokenizer_args = Args()
            epi = epitran.Epitran('eng-Latn')
            feature_extractor = PassageTokenizer(
                IpaTokenizer(), sentence_delimiter="~", sentence_preprocessor=ipa_preprocessor
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", scipy.stats.PearsonRConstantInputWarning)  # Ignore the very many PearsonRCoefficient warnings
                raw_scores, scores = score(model, feature_extractor, seed=seed)

            result = PereiraResult(
                seed=seed,
                scores=scores,
                raw_scores=raw_scores,
                model_args=model_args,
                tokenizer_args=tokenizer_args
            )
            results.append(result)

        result_set = PereiraResultSet(results=results)
        try:
            with open(gpt2_ipa_char_tokenizer_result, "wb") as f:
                pickle.dump(result_set, f)
        except OSError:
            log.warning("Warning: could not write result to file", exc_info=True)

    for result in result_set.results:
        scores = result.scores
        normalized_scores = normalize_scores(scores)
        print(
            f"Seed: {result.seed}",
            f"Scores: {scores}",
            f"Normalized scores: {normalized_scores}"
        )


if __name__ == "__main__":
    main()