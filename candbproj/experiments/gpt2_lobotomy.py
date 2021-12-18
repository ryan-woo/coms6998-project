import logging
import pickle
from pathlib import Path

import warnings
from transformers import GPT2TokenizerFast
import scipy

from candbproj import util
from candbproj.score import score, normalize_scores
from candbproj.models.modified_gpt2 import ModifiedGPT2Config, ModifiedGPT2Model
from candbproj.feature_extractors import PassageTokenizer
from candbproj.result import PereiraResult, PereiraResultSet, Args

logging.basicConfig()
log = logging.getLogger()

VARIANTS = [
    ("no_pos_embed", {"disable_pos_embeds": True}),
    ("no_res_conns", {"disable_res_conns": True}),
    ("no_scaled_attn", {"scale_attn_weights": False}),
    ("disable_mlp", {"disable_mlp": True}),
    ("disable_query_proj", {"disable_query_proj": True}),
    ("disable_key_proj", {"disable_key_proj": True}),
    ("disable_value_proj", {"disable_value_proj": True}),
    ("disable_attn_proj", {"disable_attn_proj": True}),
    ("disable_layer_norm", {"disable_layer_norm": True}),
    ("n_inner_up", {"n_inner": 8 * 768}),   # defaults to 4 * 768
    ("n_inner_same", {"n_inner": 768}),     # defaults to 4 * 768
    ("n_inner_down", {"n_inner": 385})      # defaults to 4 * 768
]

def main():
    args = util.parse_args()

    for variant, config_props in VARIANTS:
        print("evaluating %s..." % variant)
        gpt2_lobotomy_result_path = Path(__file__).parent.resolve() / f"../../results/gpt2_{variant}_lobotomy_result.pkl"
        if gpt2_lobotomy_result_path.exists():
            with open(gpt2_lobotomy_result_path, "rb") as f:
                result_set = pickle.load(f)
        else:
            results = []
            for seed in range(0, 10000, int(10000/args.n)):
                util.seeder(seed)

                model_config = ModifiedGPT2Config.from_pretrained("gpt2")
                model_config.output_hidden_states = True
                if 'n_inner' not in variant:
                    for k, v in config_props.items():
                        assert model_config.to_dict()[k] is not v
                model_config.update(config_props)

                model = ModifiedGPT2Model(model_config)
                tokenizer_args = Args(
                    args=("gpt2",)
                )
                tokenizer = GPT2TokenizerFast.from_pretrained(*tokenizer_args.args)
                feature_extractor = PassageTokenizer(tokenizer)
                model = model.eval()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", scipy.stats.PearsonRConstantInputWarning)  # Ignore the very many PearsonRCoefficient warnings
                    raw_scores, scores = score(model, feature_extractor, seed=seed)

                result = PereiraResult(
                    seed=seed,
                    raw_scores=raw_scores,
                    scores=scores,
                    model_config=model_config,
                    tokenizer_args=tokenizer_args
                )
                results.append(result)

            result_set = PereiraResultSet(results=results)
            try:
                with open(gpt2_lobotomy_result_path, "wb") as f:
                    pickle.dump(result_set, f)
            except OSError:
                log.warning("Warning: could not write result to file", exc_info=True)

        for result in result_set.results:
            scores = result.scores
            normalized_scores = normalize_scores(scores)

            print(
                f"Variant: {variant}",
                f"Seed: {result.seed}",
                f"Scores: {scores}",
                f"Normalized scores: {normalized_scores}"
            )


if __name__ == "__main__":
    main()