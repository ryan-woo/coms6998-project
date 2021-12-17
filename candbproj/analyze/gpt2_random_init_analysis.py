import pickle

import matplotlib.pyplot as plt
import re

from candbproj.result import PereiraResultSet, PereiraResult
from candbproj.analyze.analysis import extract_normalize_process, \
    df_by_key, dfs_by_layer, layer_fill_axis, key_fill_axis, \
    get_untrained_data


def group_mapper(result: PereiraResult) -> str:
    return re.sub("GPT2Model", "", result.metadata["model_name"])


def main():
    gpt2_random_init_result_path = "../../results/gpt2_random_init_result.pkl"
    with open(gpt2_random_init_result_path, "rb") as f:
        result_set: PereiraResultSet = pickle.load(f)

    normalized_random_init_scores = extract_normalize_process(
        result_set, group_mapping=group_mapper)

    normalized_random_init_scores["DefaultInit"] = get_untrained_data()
    figure, axis = plt.subplots(1, 2, figsize=(12, 6))
    dfs = dfs_by_layer(normalized_random_init_scores)
    labels = [key for key in normalized_random_init_scores.keys()]
    layer_fill_axis(axis[0], dfs, labels)

    # In this case we only care about the last layer's scores
    final_results = {}
    for initialization, result_pair in normalized_random_init_scores.items():
        final_results[initialization] = [result_pair[0][-1], result_pair[1][-1]]

    key_name = "Random Initialization"
    df = df_by_key(final_results, key_name)
    key_fill_axis(axis[1], df, key_name)
    plt.show()


if __name__ == "__main__":
    main()