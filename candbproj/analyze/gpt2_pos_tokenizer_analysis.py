import pickle

import matplotlib.pyplot as plt

from candbproj.result import PereiraResultSet, PereiraResult
from candbproj.analyze.analysis import extract_normalize_process, \
    df_by_key, dfs_by_layer, layer_fill_axis, key_fill_axis, \
    get_untrained_data
from candbproj.experiments.gpt2_pos_tokenizer import TOKENS


def tokenizer_group_mapper(name: str) -> str:
    return f"{name}Tokenizer"


def main():

    normalized_tokenizer_scores = {}
    for token_id in TOKENS.keys():
        char_tokenizer_result_path = f"../../results/gpt2_{token_id}_pos_result.pkl"
        with open(char_tokenizer_result_path, "rb") as f:
            result_set: PereiraResultSet = pickle.load(f)

        normalized_tokenizer_scores.update(extract_normalize_process(
            result_set, group_mapping=lambda _: tokenizer_group_mapper(token_id)))

    normalized_tokenizer_scores["DefaultTokenizer"] = get_untrained_data()

    figure, axis = plt.subplots(1, 2, figsize=(16, 8))
    dfs = dfs_by_layer(normalized_tokenizer_scores)
    labels = [key for key in normalized_tokenizer_scores.keys()]
    layer_fill_axis(axis[0], dfs, labels)

    # In this case we only care about the last layer's scores
    final_results = {}
    for tokenizer, result_pair in normalized_tokenizer_scores.items():
        final_results[tokenizer] = [result_pair[0][-1], result_pair[1][-1]]

    key_name = "Tokenizer"
    df = df_by_key(final_results, key_name)
    key_fill_axis(axis[1], df, key_name)
    plt.show()


if __name__ == "__main__":
    main()