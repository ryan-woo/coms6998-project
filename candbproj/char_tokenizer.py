import logging
from pathlib import Path
import pickle
import warnings
import re

from tokenizers.models import WordLevel
import scipy

from tokenizers import Tokenizer
from transformers import GPT2Model, GPT2Tokenizer
from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import FeatureExtractor
from candbproj.result import PereiraResult, Args


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


class WhitespaceReplacedPassageTokenizer(FeatureExtractor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_input_features(self, stimulus_ids, sentences):
        assert len(stimulus_ids) == len(sentences)

        # Here is the different :)
        sentences = [re.sub("\s", "~", sentence) for sentence in sentences]

        stimulus_ends = []
        length_so_far = 0
        for sentence in sentences:
            length_so_far += len(sentence)
            stimulus_ends.append(length_so_far - 1)

            # we'll join the sentences with spaces
            length_so_far += 1

        tokenized = self.tokenizer(
            ["~".join(sentences)],
            add_special_tokens=True,
            return_tensors='pt',
        )

        # note that the ending character here is usually a period
        # (we can experiment w/ the last word by subtracting 1)
        output_coords = [
            (0, stimulus_end) for stimulus_end in stimulus_ends
        ]

        return tokenized, output_coords


def main():

    chartok_dir = Path(__file__).parent.resolve() / "../custom_tokenizers/chartok"
    # chartok_dir = Path("chartok")
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
            result = pickle.load(f)
    else:
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
        feature_extractor = WhitespaceReplacedPassageTokenizer(tokenizer)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.stats.PearsonRConstantInputWarning)  # Ignore the very many PearsonRCoefficient warnings
            scores = score(model, feature_extractor)

        result = PereiraResult(
            scores=scores,
            model_args=model_args,
            tokenizer_args=tokenizer_args
        )
        try:
            with open(gpt2_char_tokenizer_result, "wb") as f:
                pickle.dump(result, f)
        except OSError:
            log.warning("Warning: could not write result to file", exc_info=True)

    scores = result.scores
    normalized_scores = normalize_scores(scores)
    print(scores, normalized_scores)


if __name__ == "__main__":
    main()