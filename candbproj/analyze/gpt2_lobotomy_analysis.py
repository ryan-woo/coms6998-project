import pickle

import matplotlib.pyplot as plt

from candbproj.result import PereiraResultSet, PereiraResult
from candbproj.analyze.analysis import extract_normalize_process, \
    df_by_key, dfs_by_layer, layer_fill_axis, key_fill_axis, \
    get_untrained_data
from candbproj.experiments.gpt2_lobotomy import VARIANTS


def group_mapper(name: str) -> str:
    return f"{name}Procedure"


def main():

    normalized_lobotomy_scores = {}
    for variant in VARIANTS:
        name = variant[0]
        result_path = f"../../results/gpt2_{name}_lobotomy_result.pkl"
        with open(result_path, "rb") as f:
            result_set: PereiraResultSet = pickle.load(f)

        normalized_lobotomy_scores.update(extract_normalize_process(
            result_set, group_mapping=lambda _: group_mapper(name)))

    normalized_lobotomy_scores["DefaultModel"] = get_untrained_data()

    figure, axis = plt.subplots(1, 2, figsize=(16, 8))
    dfs = dfs_by_layer(normalized_lobotomy_scores)
    labels = [key for key in normalized_lobotomy_scores.keys()]
    layer_fill_axis(axis[0], dfs, labels)
    axis[0].legend(loc="lower center", bbox_to_anchor=(0.5, -1.2))

    # In this case we only care about the last layer's scores
    final_results = {}
    for lobotomy_procedure, result_pair in normalized_lobotomy_scores.items():
        final_results[lobotomy_procedure] = [result_pair[0][-1], result_pair[1][-1]]

    key_name = "Procedure"
    df = df_by_key(final_results, key_name)
    key_fill_axis(axis[1], df, key_name)
    plt.subplots_adjust(bottom=0.5)
    plt.show()


if __name__ == "__main__":
    main()