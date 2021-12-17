import pickle
import warnings
import logging
from pathlib import Path

import scipy

from torch import nn
from transformers import GPT2Model, GPT2Config

from candbproj import util
from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import MFCCPassageExtractor, NUM_MFCCS
from candbproj.result import PereiraResult, PereiraResultSet, Args

logging.basicConfig()
log = logging.getLogger()

class ProjectingGPT2(nn.Module):
    def __init__(self, projection_dim, gpt2):
        super().__init__()
        self.projection_dim = projection_dim
        if self.projection_dim is not None:
            self.projector = nn.Linear(NUM_MFCCS, projection_dim)
        self.gpt2 = gpt2

    def forward(self, inputs_embeds, attention_mask):
        if self.projection_dim:
            inputs_embeds = self.projector(inputs_embeds)
        return self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)


def main():
    args = util.parse_args()

    gpt2_embedding_result_path = Path(__file__).parent.resolve() / f"../../results/gpt2_mfcc_embeddings_result.pkl"
    if gpt2_embedding_result_path.exists():
        with open(gpt2_embedding_result_path, "rb") as f:
            result_set = pickle.load(f)
    else:
        results = []
        for n_embd in [NUM_MFCCS, 300, 768]:
            for seed in range(0, 10000, int(10000/args.n)):
                util.seeder(seed)

                model_config = GPT2Config.from_pretrained("gpt2")
                assert model_config.vocab_size == 50257
                model_config.vocab_size = 0
                assert model_config.n_positions == 1024
                model_config.n_positions = 3000
                assert model_config.n_embd == 768
                model_config.n_embd = n_embd
                model_config.output_hidden_states = True

                if n_embd == NUM_MFCCS:
                    projection_dim = None
                else:
                    projection_dim = n_embd
                model = ProjectingGPT2(projection_dim, GPT2Model(model_config))
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