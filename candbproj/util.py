
from neural_nlp.benchmarks import benchmark_pool


def get_pereira():
    pereira = benchmark_pool["Pereira2018-encoding"]
    data = pereira._load_assembly(version="base")
    return data


def get_stimulus_passages(data):
    """
    Return the stimulus passages from the dataset.
    """

    stimulus_set = data.attrs["stimulus_set"]
    stimulus_set.loc[:, "passage_id"] = stimulus_set["experiment"] + stimulus_set["passage_index"].astype(str)
    return stimulus_set