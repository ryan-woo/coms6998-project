import pickle
from typing import Callable

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from candbproj.result import PereiraResultSet
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


def details_across_results(grouped_results):
    """
    Return the mean, variance across grouped_results. Returns a dictionary with the same keys
    as grouped_results, but with tuples of the (mean, variance) of the scores of each key.
    """

    details = {}
    for key, result_set in grouped_results.items():
        scores = []
        for result in result_set.results:
            scores.append(result.scores)
        results_mean = np.mean(scores, axis=0)
        results_var = np.var(scores, axis=0)

        details[key] = (results_mean, results_var)
    return details


def normalize_across_grouped_results(details):

    normalized = {}
    for key, stats in details.items():
        mean, var = stats
        normalized_mean = normalize_scores(mean)
        normalized_var = normalize_scores(var)

        normalized[key] = (normalized_mean, normalized_var)
    return normalized


def extract_process_normalize(result_set, group_mapping):
    grouped_result = group_results(result_set, group_mapping)
    details = details_across_results(grouped_result)
    normalized = normalize_across_grouped_results(details)

    return normalized


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


def key_fill_axis(axis, df, label=None):
    axis.set_title("Normalized scores by number of heads")
    axis.errorbar(
        df["n_head"],
        df["normalized_score"],
        yerr=df["variance"],
        fmt=".k",
        label=label
    )
    axis.set_xlabel("Heads")
    axis.set_ylabel("Score")
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


def generate_color():
    for r in mcolors.BASE_COLORS:

        yield r
# Don't think this will necessarily be valuable
# def scatterplot(x, y, title="Normalized scores"):
#     figure, axis = plt.subplots(1, 1, figsize=(8, 6))
#     axis.set_title("Normalized scores by number of heads")
#     sns.scatterplot(x=df["n_head"], y=df["normalized_score"], ax=axis)
#
#