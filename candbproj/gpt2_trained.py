


import warnings
from transformers import GPT2TokenizerFast, GPT2Model

from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import PassageTokenizer

def main():

    model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = model.eval()
    feature_extractor = PassageTokenizer(tokenizer)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore the very many PearsonRCoefficient warnings
        scores = score(model, feature_extractor)
    normalized_scores = normalize_scores(scores)
    print(scores, normalized_scores)

if __name__ == "__main__":
    main()