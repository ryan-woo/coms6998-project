import logging
from pathlib import Path
import pickle
import warnings
import re

from tokenizers.models import WordLevel
import scipy

from tokenizers import Tokenizer
from transformers import GPT2Model, GPT2Tokenizer
from candbproj import util
from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import PassageTokenizer
from candbproj.result import PereiraResult, PereiraResultSet, Args


logging.basicConfig()
log = logging.getLogger()


def create_char_tokenizer():
    """
    Create a custom WordLevel tokenizer with a vocabulary of
    every character used in english minus whitespace.

    The tokenizer will fail to tokenize whitespace.
    As such it is required to pre-process input strings by replacing
    whitespace with the special character ~.
    :return: Tokenizer
    """

    english_chars = {str(chr(i)): i for i in range(256)}
    vocab = {
        "<unk>": 256  # Unknown token
    }
    vocab.update(english_chars)
    tokenizer = Tokenizer(WordLevel(
        vocab=vocab,
        unk_token="<unk>",
    ))
    return tokenizer


class CharTokenizerWrapper:
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

    chartok_dir = Path(__file__).parent.resolve() / "../custom_tokenizers/chartok"
    if not chartok_dir.is_dir():
        # We must create the tokenizer from scratch
        chartok_dir.mkdir()
        tokenizer = create_char_tokenizer()
        tokenizer.model.save(str(chartok_dir))
        # Because the tokenizer is WordLevel, it does not create a merges.txt normally.
        # We create an empty merges.txt, which forces GPT2Tokenizers to effectively
        # use the tokenizer as a character-level tokenizer.
        open(chartok_dir / "merges.txt", "w").close()


    gpt2_char_tokenizer_result = Path(__file__).parent.resolve() / "../results/gpt2_char_tokenizer_result.pkl"
    if gpt2_char_tokenizer_result.exists():
        with open(gpt2_char_tokenizer_result, "rb") as f:
            result_set = pickle.load(f)
    else:
        results = []
        for seed in range(0, 10000, int(10000/args.n)):
            util.seeder(seed)

            model_args = Args(
                args = ("gpt2",),
                kwargs = {"output_hidden_states": True}
            )
            model = GPT2Model.from_pretrained(*model_args.args, **model_args.kwargs)
            model = model.eval()
            tokenizer_args = Args(
                args=(str(chartok_dir),)
            )
            tokenizer = GPT2Tokenizer.from_pretrained(*tokenizer_args.args)
            tokenizer = CharTokenizerWrapper(tokenizer)
            feature_extractor = PassageTokenizer(
                tokenizer,
                sentence_delimiter="~",
                sentence_preprocessor=lambda x: re.sub("\s", "~", x)
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
            with open(gpt2_char_tokenizer_result, "wb") as f:
                pickle.dump(result_set, f)
        except OSError:
            log.warning("Warning: could not write result to file", exc_info=True)

    for result in result_set.results:
        scores = result.scores
        normalized_scores = normalize_scores(scores)
        print(f"Seed: {result.seed}",
              f"Scores: {scores}",
              f"Normalized scores: {normalized_scores}"
              )


if __name__ == "__main__":
    main()