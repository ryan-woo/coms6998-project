import torch
import numpy as np
from tqdm import tqdm

from transformers import PreTrainedTokenizerBase, GPT2Model

from neural_nlp.benchmarks import benchmark_pool
from neural_nlp.stimuli import StimulusSet
from brainio_base.assemblies import NeuroidAssembly

from candbproj import feature_extractors


def seeder(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def get_pereira():
    pereira = benchmark_pool["Pereira2018-encoding"]
    data = pereira._load_assembly(version="base")
    return data


def get_stimulus_passages(data: NeuroidAssembly) -> StimulusSet:
    """
    Return the stimulus passages from the dataset.
    """

    stimulus_set = data.attrs["stimulus_set"]
    stimulus_set.loc[:, "passage_id"] = stimulus_set["experiment"] + stimulus_set["passage_index"].astype(str)
    return stimulus_set


def extract_activations(
    stimulus_set: StimulusSet,
    model: GPT2Model,
    feature_extractor: feature_extractors.FeatureExtractor
):
    # from stimulus_id -> # layers x 768 tensor (final representations from each layer)
    activations = {}
    for story in tqdm(sorted(set(stimulus_set['passage_id'].values))):
        story_stimuli = stimulus_set[stimulus_set['passage_id'] == story]

        sentences = []
        stimulus_ids = []
        for _, stimulus in story_stimuli.sort_values(by='sentence_num', ascending=True).iterrows():
            sentences.append(stimulus['sentence'])
            stimulus_ids.append(stimulus['stimulus_id'])

        with torch.no_grad():
            stimulus_input_features, stimulus_output_coords = \
                feature_extractor.extract_input_features(
                    stimulus_ids, sentences
                )

            output = model(**stimulus_input_features)

            for stimulus_id, (stimulus_row, stimulus_col) in zip(stimulus_ids, stimulus_output_coords):
                assert stimulus_id not in activations

                activations[stimulus_id] = torch.stack([
                    output.hidden_states[i][stimulus_row][stimulus_col] for i in range(len(output.hidden_states))
                ])
    return activations


def fold_average(experiment_pearsonrs):
    """
    take the mean of the pearsonrs across each of the random splits (across dim 0)
    """
    fold_average = {
        experiment: experiment_pearsonrs[experiment].mean(axis=0)
        for experiment in experiment_pearsonrs
    }
    return fold_average


def mean_across_experiments(experiment_voxel_ids, fold_average):
    voxel_ids = set()
    layers = fold_average["384sentences"].shape[0]
    for experiment in experiment_voxel_ids:
        voxel_ids.update(experiment_voxel_ids[experiment])
    voxel_idxs = {voxel_id: idx for idx, voxel_id in enumerate(sorted(voxel_ids))}
    shared_voxel_ids = set.intersection(*[set(vi) for vi in experiment_voxel_ids.values()])

    experiment_average = np.zeros((layers, len(voxel_ids)))
    for experiment in fold_average:
        for experiment_idx, voxel_id in enumerate(sorted(experiment_voxel_ids[experiment])):
            if voxel_id in shared_voxel_ids:
                scalar = (1 / len(fold_average.keys()))
            else:
                scalar = 1
            shared_idx = voxel_idxs[voxel_id]
            experiment_average[:, shared_idx] += scalar * fold_average[experiment][:, experiment_idx]
    return experiment_average, voxel_idxs
