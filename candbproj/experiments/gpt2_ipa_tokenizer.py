import logging
from pathlib import Path
import pickle
import warnings
import re

import scipy
import epitran
from tokenizers.models import WordLevel

from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

from candbproj import util
from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import PassageTokenizer
from candbproj.result import PereiraResult, PereiraResultSet, Args


logging.basicConfig()
log = logging.getLogger()

# IPA characters which appear in Pereira et al.

IPA_VOCAB = {
    "~": 0,
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
    "<unk>": 50
}

def create_ipa_char_tokenizer():
    """
    Create a custom WordLevel tokenizer with a vocabulary of
    every character used in Unicode minus whitespace.

    The tokenizer will fail to tokenize whitespace.
    As such it is required to pre-process input strings by replacing
    whitespace with the special character ~.
    :return: Tokenizer
    """

    unicode_chars = IPA_VOCAB
    vocab = {
        "<unk>": len(unicode_chars)  # unknown token
    }
    vocab.update(unicode_chars)
    tokenizer = Tokenizer(WordLevel(
        vocab=vocab,
        unk_token="<unk>",
    ))
    return tokenizer


class IpaCharTokenizerWrapper:
    """Required to be able to use a custom defined char_to_token() function
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        result = self.tokenizer(*args, **kwargs)

        def char_to_token(index):
            return index

        result.char_to_token = char_to_token
        return result


def main():

    args = util.parse_args()

    ipa_chartok_dir = Path(__file__).parent.resolve() / "../../custom_tokenizers/ipa"
    if not ipa_chartok_dir.is_dir():
        # We must create the tokenizer from scratch
        ipa_chartok_dir.mkdir()
        tokenizer = create_ipa_char_tokenizer()
        tokenizer.model.save(str(ipa_chartok_dir))
        # Because the tokenizer is WordLevel, it does not create a merges.txt normally.
        # We create an empty merges.txt, which forces GPT2Tokenizers to effectively
        # use the tokenizer as a character-level tokenizer.
        open(ipa_chartok_dir / "merges.txt", "w").close()


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

            tokenizer_args = Args(
                args=(str(ipa_chartok_dir),)
            )
            tokenizer = GPT2Tokenizer.from_pretrained(*tokenizer_args.args)
            tokenizer = IpaCharTokenizerWrapper(tokenizer)
            epi = epitran.Epitran('eng-Latn')
            feature_extractor = PassageTokenizer(
                tokenizer,
                sentence_delimiter="~",
                sentence_preprocessor=lambda x: re.sub("\s", "~", epi.transliterate(x))
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