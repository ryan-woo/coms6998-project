
import logging
import pickle
from pathlib import Path

import warnings
from transformers import GPT2TokenizerFast, GPT2Model, GPT2Config
import scipy


from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import PassageTokenizer
from candbproj.result import PereiraResult, Args

logging.basicConfig()
log = logging.getLogger()


def main():

    gpt2_trained_result = Path(__file__).parent.resolve() / "../results/gpt2_varied_embeddings_result.pkl"
    if gpt2_trained_result.exists():
        with open(gpt2_trained_result, "rb") as f:
            results = pickle.load(f)
    else:
        results = []
        for embeddings in [10, 20, 50, 100, 768, 1000]:

            model_config = GPT2Config.from_pretrained("gpt2")
            model_config.n_embed = embeddings
            model_config.output_hidden_states = True

            model = GPT2Model(model_config)
            tokenizer_args = Args(
                args=("gpt2",)
            )
            tokenizer = GPT2TokenizerFast.from_pretrained(*tokenizer_args.args)
            feature_extractor = PassageTokenizer(tokenizer)
            model = model.eval()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", scipy.stats.PearsonRConstantInputWarning)  # Ignore the very many PearsonRCoefficient warnings
                scores = score(model, feature_extractor)

            result = PereiraResult(
                scores=scores,
                model_config=model_config,
                tokenizer_args=tokenizer_args
            )
            results.append(result)

        try:
            with open(gpt2_trained_result, "wb") as f:
                pickle.dump(results, f)
        except OSError:
            log.warning("Warning: could not write result to file", exc_info=True)

    for result in results:
        scores = result.scores
        normalized_scores = normalize_scores(scores)


        print(f"embeddings {result.model_config.n_embed}", scores, normalized_scores)


if __name__ == "__main__":
    main()