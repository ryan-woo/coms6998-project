import logging
import pickle
from pathlib import Path

import warnings
from transformers import GPT2TokenizerFast, GPT2Model, GPT2Config
import torch.nn as nn
from transformers.modeling_utils import Conv1D
import scipy

from candbproj import util
from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import PassageTokenizer
from candbproj.result import PereiraResult, PereiraResultSet, Args



logging.basicConfig()
log = logging.getLogger()


"""
Definition of init_weights is found in the docs: https://huggingface.co/transformers/v1.1.0/_modules/pytorch_transformers/modeling_gpt2.html
￼
￼    def init_weights(self, module):
￼        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
￼            # Slightly different from the TF version which uses truncated_normal for initialization
￼            # cf https://github.com/pytorch/pytorch/pull/5617
￼            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
￼            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
￼                module.bias.data.zero_()
￼        elif isinstance(module, LayerNorm):
￼            module.bias.data.zero_()
￼            module.weight.data.fill_(1.0)
￼
Helpful:
https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch/49433937#49433937
"""


class XavierNormalInitGPT2Model(GPT2Model):

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.xavier_normal_(module.weight)
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class XavierUniformInitGPT2Model(GPT2Model):

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.xavier_uniform_(module.weight)
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class HeInitGPT2Model(GPT2Model):
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):

            nn.init.kaiming_uniform_(module.weight)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def main():

    args = util.parse_args()

    gpt2_model_result_path = Path(__file__).parent.resolve() / f"../../results/gpt2_random_init_result.pkl"

    if gpt2_model_result_path.exists():
        with open(gpt2_model_result_path, "rb") as f:
            result_set = pickle.load(f)

    else:
        model_classes = (XavierUniformInitGPT2Model, XavierNormalInitGPT2Model, HeInitGPT2Model)
        results = []
        for model_class in model_classes:
            for seed in range(0, 10000, int(10000/args.n)):
                util.seeder(seed)

                model_config = GPT2Config.from_pretrained("gpt2")
                model_config.output_hidden_states = True
                model = model_class(model_config)
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
                    scores=scores,
                    raw_scores=raw_scores,
                    model_config=model_config,
                    tokenizer_args=tokenizer_args,
                    metadata={"model_name": model_class.__name__}
                )

                results.append(result)

        result_set = PereiraResultSet(results=results)
        try:
            with open(gpt2_model_result_path, "wb") as f:
                pickle.dump(result_set, f)
        except OSError:
            log.warning("Warning: could not write result to file", exc_info=True)

    for result in result_set.results:
        scores = result.scores
        normalized_scores = normalize_scores(scores)
        model_name = result.metadata.get("model_name", "Unknown model")

        print(f"Seed: {result.seed}",
              f"model name: {model_name}",
              f"Scores: {scores}",
              f"Normalized scores: {normalized_scores}"
              )


if __name__ == "__main__":
    main()