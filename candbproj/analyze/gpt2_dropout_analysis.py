import pickle

import matplotlib.pyplot as plt

from candbproj.result import PereiraResultSet, PereiraResult
from candbproj.analyze.analysis import extract_normalize_process, \
    df_by_key, dfs_by_layer, layer_fill_axis, key_fill_axis, \
    get_untrained_data
from candbproj.experiments.gpt2_dropout_variations import ALL_DROPOUT_LOCATIONS


def group_mapper(result: PereiraResult, dropout_location: str) -> str:
    return f"{dropout_location}{result.model_config.to_dict()[dropout_location]}"


def main():

    normalized_lobotomy_scores = {}
    for dropout_location in ALL_DROPOUT_LOCATIONS:
        result_path = f"../../results/gpt2_varied_{dropout_location}_dropout_result.pkl"
        with open(result_path, "rb") as f:
            result_set: PereiraResultSet = pickle.load(f)

        normalized_lobotomy_scores.update(extract_normalize_process(
            result_set, group_mapping=lambda r: group_mapper(r, dropout_location)))

    normalized_lobotomy_scores["DefaultModel"] = get_untrained_data()

    figure, axis = plt.subplots(1, 2, figsize=(16, 8))
    dfs = dfs_by_layer(normalized_lobotomy_scores)
    labels = [key for key in normalized_lobotomy_scores.keys()]
    layer_fill_axis(axis[0], dfs, labels)
    axis[0].legend(loc="center right", bbox_to_anchor=(1.35, 0.5))

    # In this case we only care about the last layer's scores
    final_results = {}
    for lobotomy_procedure, result_pair in normalized_lobotomy_scores.items():
        final_results[lobotomy_procedure] = [result_pair[0][-1], result_pair[1][-1]]

    key_name = "Procedure"
    df = df_by_key(final_results, key_name)
    key_fill_axis(axis[1], df, key_name)
    plt.subplots_adjust(bottom=0.2, wspace=0.5)
    plt.show()


if __name__ == "__main__":
    main()