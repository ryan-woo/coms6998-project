import logging
import pickle
from pathlib import Path

import warnings
from transformers import GPT2TokenizerFast, GPT2Model, GPT2Config
import scipy

from candbproj import util
from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import PassageTokenizer
from candbproj.result import PereiraResult, PereiraResultSet, Args

logging.basicConfig()
log = logging.getLogger()

ALL_DROPOUT_LOCATIONS = [
    'attn_pdrop',
    'embd_pdrop',
    'resid_pdrop'
]

def main():
    args = util.parse_args()

    for dropout_location in ALL_DROPOUT_LOCATIONS:
        gpt2_dropout_result_path = Path(__file__).parent.resolve() / f"../../results/gpt2_varied_{dropout_location}_dropout_result.pkl"
        if gpt2_dropout_result_path.exists():
            with open(gpt2_dropout_result_path, "rb") as f:
                result_set = pickle.load(f)
        else:
            results = []
            for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for seed in range(0, 10000, int(10000/args.n)):
                    util.seeder(seed)

                    model_config = GPT2Config.from_pretrained("gpt2")
                    model_config.output_hidden_states = True

                    assert model_config.to_dict()[dropout_location] is not None
                    model_config.update({dropout_location: dropout})

                    # manually disable other dropout locations
                    for other_dropout_location in ALL_DROPOUT_LOCATIONS:
                        if other_dropout_location != dropout_location:
                            assert model_config.to_dict()[dropout_location] is not None
                            model_config.update({other_dropout_location: 0.0})

                    model = GPT2Model(model_config)
                    tokenizer_args = Args(
                        args=("gpt2",)
                    )
                    tokenizer = GPT2TokenizerFast.from_pretrained(*tokenizer_args.args)
                    feature_extractor = PassageTokenizer(tokenizer)
                    # NOTE: we want the dropout to fire so we don't put the model in eval mode!  
                    # this should randomly sparsify the network (in a deterministic / consistent way 
                    # for each back of examples because we manually seed torch before calling #forward)
                    model = model.train()

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", scipy.stats.PearsonRConstantInputWarning)  # Ignore the very many PearsonRCoefficient warnings
                        raw_scores, scores = score(
                            model, feature_extractor, seed=seed, dropout_seed=seed
                        )

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
                with open(gpt2_dropout_result_path, "wb") as f:
                    pickle.dump(result_set, f)
            except OSError:
                log.warning("Warning: could not write result to file", exc_info=True)

        for result in result_set.results:
            scores = result.scores
            normalized_scores = normalize_scores(scores)

            print(f"Seed: {result.seed}",
                  f"Location: {dropout_location}",
                  f"Dropout: {result.model_config.to_dict()[dropout_location]}",
                  f"Scores: {scores}",
                  f"Normalized scores: {normalized_scores}"
                  )


if __name__ == "__main__":
    main()