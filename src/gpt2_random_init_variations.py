import neural_nlp
from neural_nlp import score
from neural_nlp.models import model_pool, model_layers, implementations
from brainscore.utils import LazyLoad
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
import torch.nn as nn
from transformers.modeling_utils import Conv1D
from transformers.modeling_bert import BertLayerNorm as LayerNorm



"""
Definition of init_weights is found in the docs: https://huggingface.co/transformers/v1.1.0/_modules/pytorch_transformers/modeling_gpt2.html

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
            nn.init.xavier_normal(module.weight)
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class XavierUniformInitGPT2Model(GPT2Model):

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            torch.nn.init.xavier_uniform_(module.weight)
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
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
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# Use different embeddings (dimensions) for gpt-2 layers. Consistently use 24 layers
def untrained_random_init_gpt2_test(model_class: GPT2Model):
    n_layer = 12
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2", output_hidden_states=True)
    config.n_layer = n_layer
    model = model_class(config)

    wrapper = implementations._PytorchTransformerWrapper(
        identifier=f"untrained-{model_class}-embedding-gpt2-test",
        tokenizer = tokenizer,
        model=model,
        tokenizer_special_tokens=('ġ',),
        layers=list(("drop",) + tuple(f"encoder.h.{i}" for i in range(n_layer))),
        sentence_average=implementations.word_last
    )
    return wrapper

def main():
    n_layer = 12

    for model_class in [XavierInitGPT2Model, XavierUniformInitGPT2Model, HeInitGPT2Model]:
        model_pool[f"untrained-{model_class}-embedding-gpt2-test"] = (
            lambda e: LazyLoad(
                lambda: untrained_random_init_gpt2_test(e)
            )
        )(model_class)
        model_layers[f"untrained-{model_class}-embedding-gpt2-test"] = list(
            ("drop",) + tuple(f"encoder.h.{i}" for i in range(n_layer))
        )

    for model_class in [XavierInitGPT2Model, XavierUniformInitGPT2Model, HeInitGPT2Model]:
        score(benchmark="Pereira2018-encoding", model=f"untrained-{model_class}-embedding-gpt2-test")

if __name__ == "__main__":
    main()