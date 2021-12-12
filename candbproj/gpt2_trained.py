


import warnings
from transformers import GPT2TokenizerFast, GPT2Model

from candbproj.score import score, normalize_scores

def main():

    model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = model.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore the very many PearsonRCoefficient warnings
        scores = score(model, tokenizer)
    normalized_scores = normalize_scores(scores)
    print(scores, normalized_scores)

if __name__ == "__main__":
    main()