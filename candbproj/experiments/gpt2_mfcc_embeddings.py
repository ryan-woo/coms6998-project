import logging
import pickle
from pathlib import Path

import warnings
from transformers import GPT2Model, GPT2Config
import scipy

from candbproj import util
from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import MFCCPassageExtractor, NUM_MFCCS
from candbproj.result import PereiraResult, PereiraResultSet, Args

logging.basicConfig()
log = logging.getLogger()


def main():
    args = util.parse_args()

    gpt2_embedding_result_path = Path(__file__).parent.resolve() / f"../../results/gpt2_mfcc_embeddings_result.pkl"
    if gpt2_embedding_result_path.exists():
        with open(gpt2_embedding_result_path, "rb") as f:
            result_set = pickle.load(f)
    else:
        results = []
        for seed in range(0, 10000, int(10000/args.n)):
            util.seeder(seed)

            model_config = GPT2Config.from_pretrained("gpt2")
            assert model_config.vocab_size == 50257
            model_config.vocab_size = 0
            assert model_config.n_positions == 1024
            model_config.n_positions = 3000
            assert model_config.n_embd == 768
            model_config.n_embd = NUM_MFCCS
            model_config.output_hidden_states = True

            model = GPT2Model(model_config)
            feature_extractor_args = Args(
                args=("mfcc",)
            )
            feature_extractor = MFCCPassageExtractor()
            model = model.eval()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", scipy.stats.PearsonRConstantInputWarning)  # Ignore the very many PearsonRCoefficient warnings
                raw_scores, scores = score(model, feature_extractor, seed=seed)

            result = PereiraResult(
                seed=seed,
                raw_scores=raw_scores,
                scores=scores,
                model_config=model_config,
                tokenizer_args=feature_extractor_args
            )
            results.append(result)

        result_set = PereiraResultSet(results=results)
        try:
            with open(gpt2_embedding_result_path, "wb") as f:
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