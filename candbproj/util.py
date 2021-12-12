import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, GPT2Model
import numpy as np

from neural_nlp.benchmarks import benchmark_pool
from neural_nlp.stimuli import StimulusSet
from brainio_base.assemblies import NeuroidAssembly



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


def extract_passage_activations(
    stimulus_set: StimulusSet,
    model: GPT2Model,
    tokenizer: PreTrainedTokenizerBase
):
    # from stimulus_id -> 13 x 768 tensor (final representations from each layer)
    activations = {}
    for story in tqdm(sorted(set(stimulus_set['passage_id'].values))):
        story_stimuli = stimulus_set[stimulus_set['passage_id'] == story]

        sentences = []
        stimulus_ids = []
        stimulus_ends = []
        length_so_far = 0
        for _, stimulus in story_stimuli.sort_values(by='sentence_num', ascending=True).iterrows():
            length_so_far += len(stimulus['sentence'])
            sentences.append(stimulus['sentence'])
            stimulus_ids.append(stimulus['stimulus_id'])
            stimulus_ends.append(length_so_far - 1)

            # we'll join the sentences with spaces
            length_so_far += 1

        with torch.no_grad():
            tokenized = tokenizer(
                [' '.join(sentences)],
                add_special_tokens=True,
                return_tensors='pt'
            )

            # note that the ending character here is usually a period
            # (we can experiment w/ the last word by subtracting 1)
            stimulus_token_ends = [
                tokenized.char_to_token(stimulus_end) for stimulus_end in stimulus_ends
            ]

            output = model(**tokenized)

            for stimulus_id, stimulus_token_end in zip(stimulus_ids, stimulus_token_ends):
                assert stimulus_id not in activations

                # get hidden state of each final token for each stimulus

                activations[stimulus_id] = torch.stack([
                    output.hidden_states[i][0][stimulus_token_end] for i in range(len(output.hidden_states))
                ])
    return activations


def extract_sentence_activations(
    stimulus_set: StimulusSet,
    model: GPT2Model,
    tokenizer: PreTrainedTokenizerBase
):
    # from stimulus_id -> 13 x 768 tensor (final representations from each layer)
    activations = {}
    for stimulus_id in tqdm(sorted(set(stimulus_set['stimulus_id'].values))):
        stimulus = stimulus_set[stimulus_set['stimulus_id'] == stimulus_id]

        assert len(stimulus) == 1

        with torch.no_grad():
            tokenized = tokenizer(
                [stimulus.iloc[0]['sentence']],
                add_special_tokens=True,
                return_tensors='pt'
            )

            output = model(**tokenized)

            assert stimulus_id not in activations

            activations[stimulus_id] = torch.stack([
                output.hidden_states[i][0][-1] for i in range(len(output.hidden_states))
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
