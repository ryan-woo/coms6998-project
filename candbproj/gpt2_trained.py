
import logging
import pickle
from pathlib import Path

import warnings
from transformers import GPT2TokenizerFast, GPT2Model
import scipy

from candbproj import util
from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import PassageTokenizer
from candbproj.result import PereiraResult, Args

logging.basicConfig()
log = logging.getLogger()


def main():
    for seed in range(0, 200, 100):
        util.seeder(seed)

        gpt2_trained_result = Path(__file__).parent.resolve() / f"../results/gpt2_trained_{seed}_result.pkl"
        if gpt2_trained_result.exists():
            with open(gpt2_trained_result, "rb") as f:
                result = pickle.load(f)
        else:
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
