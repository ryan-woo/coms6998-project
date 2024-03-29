import pickle
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from candbproj.result import PereiraResultSet, PereiraResult
from candbproj.score import normalize_scores


def group_results(result_set: PereiraResultSet, group_mapping: Callable):
    """
    Takes a group_mapping: a callable which takes a PereiraResult and
    returns a key for the group. Results which have the same group_mapping
    will be returned in a list under the same key.

    group_mapping must return a hashable type.
    """

    grouped_results = {}
    for result in result_set.results:
        r = grouped_results.setdefault(
            group_mapping(result),
            PereiraResultSet(results=[])
        )
        r.results.append(result)

    return grouped_results


def details_across_results(normalized_results):
    """
    Return the mean, variance across grouped_results. Returns a dictionary with the same keys
    as grouped_results, but with tuples of the (mean, variance) of the scores of each key.
    """

    details = {}
    for key, scores in normalized_results.items():
        results_mean = np.mean(scores, axis=0)
        results_var = np.var(scores, axis=0)

        details[key] = (results_mean, results_var)
    return details


def normalize_across_grouped_results(grouped_result):

    normalized = {}
    for key, result_set in grouped_result.items():
        scores = []
        for result in result_set.results:
            scores.append(result.scores)
        scores = normalize_scores(scores)
        normalized[key] = scores
    return normalized


def extract_normalize_process(result_set, group_mapping):
    grouped_result = group_results(result_set, group_mapping)
    normalized = normalize_across_grouped_results(grouped_result)
    details = details_across_results(normalized)

    return details


def df_by_key(grouped_results, key_name: str):
    df = pd.DataFrame.from_dict(
        grouped_results,
        orient="index",
        columns=["normalized_score", "variance"]
    )
    df = df.reset_index()
    df = df.rename({"index": key_name}, axis="columns")
    return df


def dfs_by_layer(grouped_results):

    dfs = []
    for value in grouped_results.values():
        d = {
            "normalized_score": value[0],
            "variance": value[1],
        }
        df = pd.DataFrame(d)
        df = df.reset_index()
        df = df.rename({"index": "layer"}, axis="columns")
        dfs.append(df)
    return dfs


def key_fill_axis(axis, df, key_name, label=None):
    axis.set_title(f"Normalized scores by {key_name}")
    axis.errorbar(
        df[key_name],
        df["normalized_score"],
        yerr=df["variance"],
        fmt=".k",
        label=label
    )
    axis.set_xlabel(key_name)
    axis.set_ylabel("Score")
    plt.setp(axis.get_xticklabels(), rotation=30, horizontalalignment='right')
    if label is not None:
        axis.legend()


def layer_fill_axis(axis, dfs, labels):
    color_generator = generate_color()
    axis.set_title("Normalized scores by layer")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Score")
    for df, label in zip(dfs, labels):
        axis.errorbar(df["layer"],
                         df["normalized_score"],
                         yerr=df["variance"],
                         fmt=".",
                         c=next(color_generator),
                         label=label
                         )
    axis.legend()


def embedding_group_mapper(result: PereiraResult):
    return result.model_config.n_embd


def get_untrained_data():

    gpt2_embedding_result_path = "../../results/gpt2_varied_embeddings_result.pkl"
    with open(gpt2_embedding_result_path, "rb") as f:
        result_set: PereiraResultSet = pickle.load(f)

    normalized_embedding_scores = extract_normalize_process(result_set, group_mapping=embedding_group_mapper)

    default_initialization_result = normalized_embedding_scores[768]  # Get the default size
    return default_initialization_result


def generate_color():
    css_colors = {k:v for k, v in mcolors.CSS4_COLORS.items() if k in {'lightcoral', 'palegreen', 'magenta'}}
    colors = {**mcolors.BASE_COLORS, **mcolors.TABLEAU_COLORS, **css_colors}
    colors.pop("w")

    for r in colors:
        yield r
