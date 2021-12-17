import pickle

import matplotlib.pyplot as plt

from candbproj.result import PereiraResultSet, PereiraResult
from candbproj.analyze.analysis import extract_normalize_process, \
    df_by_key, dfs_by_layer, layer_fill_axis, key_fill_axis


def group_mapper(result: PereiraResult) -> int:
    return result.model_config.n_head


def main():
    gpt2_embedding_result_path = "../../results/gpt2_varied_attn_heads_result.pkl"
    with open(gpt2_embedding_result_path, "rb") as f:
        result_set: PereiraResultSet = pickle.load(f)

    normalized_attn_head_scores = extract_normalize_process(
        result_set, group_mapping=group_mapper)

    figure, axis = plt.subplots(1, 2, figsize=(12, 6))
    dfs = dfs_by_layer(normalized_attn_head_scores)
    labels = [f"{key} heads" for key in normalized_attn_head_scores.keys()]
    layer_fill_axis(axis[0], dfs, labels)

    # In this case we only care about the last layer's scores
    final_results = {}
    for attn_heads, result_pair in normalized_attn_head_scores.items():
        final_results[attn_heads] = [result_pair[0][-1], result_pair[1][-1]]

    key_name = "n_head"
    df = df_by_key(final_results, key_name)
    key_fill_axis(axis[1], df, key_name)
    plt.show()


if __name__ == "__main__":
    main()