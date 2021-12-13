
import logging
import pickle
from pathlib import Path

import warnings
from transformers import GPT2TokenizerFast, GPT2Model
import scipy

from candbproj.score import score, normalize_scores
from candbproj.result import PereiraResult, Args

logging.basicConfig()
log = logging.getLogger()


def main():

    gpt2_trained_result = Path(__file__).parent.resolve() / "../results/gpt2_trained_result.pkl"
    if gpt2_trained_result.exists():
        with open(gpt2_trained_result, "rb") as f:
            result = pickle.load(f)
    else:
        model_args = Args(
            args = ("gpt2",),
            kwargs = {"output_hidden_states": True}
        )
        model = GPT2Model.from_pretrained(*model_args.args, **model_args.kwargs)

        tokenizer_args = Args(
            args=("gpt2",)
        )
        tokenizer = GPT2TokenizerFast.from_pretrained(*tokenizer_args.args)
        model = model.eval()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.stats.PearsonRConstantInputWarning)  # Ignore the very many PearsonRCoefficient warnings
            scores = score(model, tokenizer)

        result = PereiraResult(
            scores=scores,
            model_args=model_args,
            tokenizer_args=tokenizer_args
        )
        try:
            with open(gpt2_trained_result, "wb") as f:
                pickle.dump(result, f)
        except OSError:
            log.warning("Warning: could not write result to file", exc_info=True)

    scores = result.scores
    normalized_scores = normalize_scores(scores)
    print(scores, normalized_scores)


if __name__ == "__main__":
    main()