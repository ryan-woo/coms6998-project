import warnings
from collections import defaultdict

from neural_nlp.benchmarks import benchmark_pool
import numpy as np
from tqdm import tqdm
from brainio_base.assemblies import NeuroidAssembly
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit

from candbproj import util


def experiment_voxel_info(data: NeuroidAssembly):
    experiment_voxels = defaultdict(list)
    experiment_voxel_ids = defaultdict(set)
    experiment_voxel_nas = defaultdict(set)
    experiment_stimuli = defaultdict(list)
    for presentation_id, stimulus_id, experiment in tqdm(zip(
            range(data.shape[0]),
            data['stimulus_id'].values,
            data['experiment'].values
    ), desc="preprocessing Pereira"):
        voxels = []
        for voxel_id, atlas in zip(range(data.shape[1]), data['atlas'].values):
            if atlas == 'language':
                experiment_voxel_ids[experiment].add(voxel_id)
                voxel = data.values[presentation_id][voxel_id]
                if np.isnan(voxel):
                    experiment_voxel_nas[experiment].add(voxel_id)
                voxels.append(voxel)

        experiment_voxels[experiment].append(voxels)
        experiment_stimuli[experiment].append(stimulus_id)

    for experiment in experiment_voxel_ids:
        experiment_voxel_ids[experiment] = list(sorted(experiment_voxel_ids[experiment]))

    return experiment_voxels, experiment_voxel_ids, experiment_voxel_nas, experiment_stimuli


def filter_na_voxels(experiment_voxels=None, experiment_voxel_ids=None, experiment_voxel_nas=None):
    """
    Filter voxels that are na from experiment_voxels
    """
    if experiment_voxels is None or experiment_voxel_ids is None or experiment_voxel_nas is None:
        raise ValueError

    experiments = {}
    for experiment in experiment_voxels:
        voxel_ids = experiment_voxel_ids[experiment]
        experiments[experiment] = np.array(
            [
                [
                    voxel for voxel_id, voxel in zip(voxel_ids, voxels)
                    if voxel_id not in experiment_voxel_nas[experiment]
                ]
                for voxels in experiment_voxels[experiment]
            ]
        )
        experiment_voxel_ids[experiment] = list(sorted(set(voxel_ids) - experiment_voxel_nas[experiment]))
    return experiments, experiment_voxel_ids


def raw_score(experiments=None, experiment_stimuli=None, activations=None, folds=5, seed=100):

    if experiments is None or experiment_stimuli is None or activations is None:
        raise ValueError

    layers = activations[list(activations.keys())[0]].shape[0]

    experiment_pearsonrs = {
        experiment: np.zeros((folds, layers, experiments[experiment].shape[1]))
        for experiment in experiments
    }

    for experiment, brain_reps in experiments.items():
        # splits need to be by stimulus_id (how do we shuffle here?)
        # (though really they should be by passage_id given how we're doing the GPT2 encoding...
        # otherwise the test set will leak into the train set...)
        # in the brain-score repo, CrossRegressedCorrelation uses a train_size of 0.9
        k_folds = GroupShuffleSplit(n_splits=folds, train_size=0.9, random_state=100)

        for fold, (train_indices, test_indices) in enumerate(
                k_folds.split(brain_reps, groups=experiment_stimuli[experiment])
        ):
            train_brain_reps, test_brain_reps = brain_reps[train_indices], brain_reps[test_indices]
            for layer_num in tqdm(range(layers), desc='%s-fold%s' % (experiment, fold)):
                train_hidden_states = np.stack([
                    activations[experiment_stimuli[experiment][brain_rep_idx]][layer_num].numpy()
                    for brain_rep_idx in train_indices
                ])
                test_hidden_states = np.stack([
                    activations[experiment_stimuli[experiment][brain_rep_idx]][layer_num].numpy()
                    for brain_rep_idx in test_indices
                ])

                # Using defaults for LinearRegression like they do in neural-nlp
                model = LinearRegression().fit(train_hidden_states, train_brain_reps)
                pred_brain_reps = model.predict(test_hidden_states)

                nans = 0
                for idx in range(test_brain_reps.shape[1]):
                    pred_voxels = pred_brain_reps[:, idx]
                    test_voxels = test_brain_reps[:, idx]
                    r, _ = pearsonr(pred_voxels, test_voxels)

                    if np.isnan(r):
                        r = 0.0
                        nans += 1
                        assert nans < 20

                    experiment_pearsonrs[experiment][fold][layer_num][idx] = r

    return experiment_pearsonrs


def score(model, feature_extractor, seed, dropout_seed=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="xarray subclass", category=FutureWarning)
        pereira_data = util.get_pereira()
        stimulus_set = util.get_stimulus_passages(pereira_data)

    activations = util.extract_activations(
        stimulus_set, model, feature_extractor, dropout_seed=None)

    experiment_voxels, experiment_voxel_ids, experiment_voxel_nas, experiment_stimuli = \
        experiment_voxel_info(pereira_data)

    experiments, experiment_voxel_ids = filter_na_voxels(
        experiment_voxels=experiment_voxels,
        experiment_voxel_ids=experiment_voxel_ids,
        experiment_voxel_nas=experiment_voxel_nas
    )
    experiment_pearsonrs = raw_score(experiments, experiment_stimuli, activations, seed=seed)
    fold_average = util.fold_average(experiment_pearsonrs)
    experiment_average, voxel_idxs = util.mean_across_experiments(experiment_voxel_ids, fold_average)

    scores = []
    subjects = {voxel_id: subject for voxel_id, subject in enumerate(pereira_data['subject'].values)}
    for layer_num in range(experiment_average.shape[0]):
        by_subject = defaultdict(list)
        for idx, voxel_id in enumerate(sorted(voxel_idxs.keys())):
            by_subject[subjects[voxel_id]].append(experiment_average[layer_num, idx])
        scores.append(np.median([np.median(subject_rs) for subject_rs in by_subject.values()]))
    return fold_average, scores

def normalize_scores(scores):
    pereira = benchmark_pool["Pereira2018-encoding"]
    scores = np.array(scores)
    ceiling = pereira.ceiling.sel(aggregation='center').item()
    return scores / ceiling