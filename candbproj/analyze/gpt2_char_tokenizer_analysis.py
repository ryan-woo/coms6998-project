import pickle

import matplotlib.pyplot as plt

from candbproj.result import PereiraResultSet, PereiraResult
from candbproj.analyze.analysis import extract_normalize_process, \
    df_by_key, dfs_by_layer, layer_fill_axis, key_fill_axis, \
    get_untrained_data


def group_mapper(name: str) -> str:
    return name


def main():
    char_tokenizer_result_path = "../../results/gpt2_char_tokenizer_result.pkl"
    with open(char_tokenizer_result_path, "rb") as f:
        result_set: PereiraResultSet = pickle.load(f)

    normalized_char_tokenizer_scores = extract_normalize_process(
        result_set, group_mapping=lambda _: group_mapper("CharTokenizer"))

    ipa_tokenizer_result_path = "../../results/gpt2_ipa_char_tokenizer_result.pkl"
    with open(ipa_tokenizer_result_path, "rb") as f:
        result_set: PereiraResultSet = pickle.load(f)
    normalized_char_tokenizer_scores.update(extract_normalize_process(
        result_set, group_mapping=lambda _: group_mapper("IPACharTokenizer")))
    normalized_char_tokenizer_scores["DefaultTokenizer"] = get_untrained_data()

    figure, axis = plt.subplots(1, 2, figsize=(12, 6))
    dfs = dfs_by_layer(normalized_char_tokenizer_scores)
    labels = [key for key in normalized_char_tokenizer_scores.keys()]
    layer_fill_axis(axis[0], dfs, labels)

    # In this case we only care about the last layer's scores
    final_results = {}
    for tokenizer, result_pair in normalized_char_tokenizer_scores.items():
        final_results[tokenizer] = [result_pair[0][-1], result_pair[1][-1]]

    key_name = "Tokenizer"
    df = df_by_key(final_results, key_name)
    key_fill_axis(axis[1], df, key_name)
    plt.show()


if __name__ == "__main__":
    main()