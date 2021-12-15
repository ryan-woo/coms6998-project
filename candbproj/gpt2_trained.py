import argparse
import logging
import pickle
from pathlib import Path

import warnings
from transformers import GPT2TokenizerFast, GPT2Model
import scipy

from candbproj import util
from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import PassageTokenizer
from candbproj.result import PereiraResult, PereiraResultSet, Args

logging.basicConfig()
log = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", type=int,
        help="The number of times to run the result. Please use a number less than 10000",
        default=10
    )

    return parser.parse_args()

def main():

    args = parse_args()

    gpt2_trained_result = Path(__file__).parent.resolve() / f"../results/gpt2_trained_result.pkl"
    if gpt2_trained_result.exists():
        with open(gpt2_trained_result, "rb") as f:
            result_set = pickle.load(f)
    else:
        results = []
        for seed in range(0, 10000, int(10000 / args.n)):
            util.seeder(seed)

            model_args = Args(
                args = ("gpt2",),
                kwargs = {"output_hidden_states": True}
            )
            model = GPT2Model.from_pretrained(*model_args.args, **model_args.kwargs)
            model = model.eval()
            tokenizer_args = Args(
                args=("gpt2",)
            )
            tokenizer = GPT2TokenizerFast.from_pretrained(*tokenizer_args.args)
            feature_extractor = PassageTokenizer(tokenizer)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", scipy.stats.PearsonRConstantInputWarning)  # Ignore the very many PearsonRCoefficient warnings
                scores = score(model, feature_extractor, seed=seed)

            result = PereiraResult(
                seed=seed,
                scores=scores,
                model_args=model_args,
                tokenizer_args=tokenizer_args
            )
            results.append(result)

        result_set = PereiraResultSet(results=results)
        try:
            with open(gpt2_trained_result, "wb") as f:
                pickle.dump(result_set, f)
        except OSError:
            log.warning("Warning: could not write result to file", exc_info=True)

    for result in result_set.results:

        scores = result.scores
        normalized_scores = normalize_scores(scores)
        print(f"Seed: {result.seed}",
              f"Scores {scores}",
              f"Normalized scores: {normalized_scores}"
              )

if __name__ == "__main__":
    main()
