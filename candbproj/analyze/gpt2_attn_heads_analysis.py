import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from candbproj.result import PereiraResultSet, PereiraResult
from candbproj.analyze.analysis import extract_process_normalize, convert_to_df


def group_mapper(result: PereiraResult) -> int:
    return result.model_config.n_head


def main():
    gpt2_embedding_result_path = "../../results/gpt2_varied_attn_heads_result.pkl"
    with open(gpt2_embedding_result_path, "rb") as f:
        result_set: PereiraResultSet = pickle.load(f)

    normalized_attn_head_scores = extract_process_normalize(result_set, group_mapping=group_mapper)

    # In this case we only care about the last layer's scores
    final_results = {}
    for attn_heads, result_pair in normalized_attn_head_scores.items():
        final_results[attn_heads] = [result_pair[0][-1], result_pair[1][-1]]

    df = convert_to_df(final_results, "n_head")

    # Plot this data with a scatterplot
    figure, axis = plt.subplots(1, 1, figsize=(8,6))
    axis.set_title("Normalized scores by number of heads")
    sns.scatterplot(x=df["n_head"], y=df["normalized_score"], ax=axis)
    plt.show()


if __name__ == "__main__":
    main()