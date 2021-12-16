import os
import pickle
import logging
import warnings
import argparse
from pathlib import Path

import spacy
import scipy
from lazy_load import lazy
from transformers import GPT2Config, GPT2TokenizerFast, GPT2Model

from candbproj import util
from candbproj.score import score, normalize_scores
from candbproj.feature_extractors import PassageTokenizer
from candbproj.result import PereiraResult, PereiraResultSet, Args

logging.basicConfig()
log = logging.getLogger()

TOKENS = {
    'coarse': [
        'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
        'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
    ],
    'fine': [
        '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'CC', 'CD', 
        'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 
        'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 
        'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
        'WDT', 'WP', 'WP$', 'WRB', 'XX', '``'
    ]
}
NER = [
    'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 
    'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 
    'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART'
]
TOKENS['coarse_pos_ner'] = TOKENS['coarse'] + NER
TOKENS['fine_pos_ner'] = TOKENS['fine'] + NER

EN = lazy(lambda: spacy.load('en_core_web_trf'))
PREPROCESSING_DIR = Path(__file__).parent.resolve() / '../preprocessed'
PARSE_CACHE_FILE = os.path.join(PREPROCESSING_DIR, 'parses.pkl')

PARSES = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", type=int,
        help="The number of times to run the result. Please use a number less than 10000",
        default=10
    )

    return parser.parse_args()

def get_parse(sentence_id):
    global PARSES

    if PARSES is None:
        if os.path.exists(PARSE_CACHE_FILE):
            with open(PARSE_CACHE_FILE, 'rb') as f:
                PARSES = pickle.load(f)
        else:
            PARSES = {}

    if sentence_id not in PARSES:
        PARSES[sentence_id] = EN(sentence)
        with open(PARSE_CACHE_FILE, 'wb') as f:
            pickle.dump(PARSES, f)

    return PARSES[sentence_id]

def get_coarse_pos_tags(sentence_id, sentence):
    return ' '.join([token.pos_ for token in get_parse(sentence_id)])

def get_fine_pos_tags(sentence_id, sentence):
    return ' '.join([token.tag_ for token in get_parse(sentence_id)])

def get_coarse_pos_ner_tags(sentence_id, sentence):
    tags = []
    for token in get_parse(sentence_id):
        if token.ent_iob_ == 'B':
            tags.append(token.ent_type_)
        elif token.ent_iob_ == 'O':
            tags.append(token.pos_)
        else:
            assert token.ent_iob_ == 'I'
    return ' '.join(tags)

def get_fine_pos_ner_tags(sentence_id, sentence):
    tags = []
    for token in get_parse(sentence_id):
        if token.ent_iob_ == 'B':
            tags.append(token.ent_type_)
        elif token.ent_iob_ == 'O':
            tags.append(token.tag_)
        else:
            assert token.ent_iob_ == 'I'
    return ' '.join(tags)

PREPROCESSORS = {
    'coarse': get_coarse_pos_tags,
    'fine': get_fine_pos_tags,
    'coarse_pos_ner': get_coarse_pos_ner_tags,
    'fine_pos_ner': get_fine_pos_ner_tags,
}

def main():
    args = parse_args()

    for granularity in sorted(TOKENS.keys()):
        gpt2_pos_result = Path(__file__).parent.resolve() / f"../../results/gpt2_{granularity}_pos_result.pkl"
        if gpt2_pos_result.exists():
            with open(gpt2_pos_result, "rb") as f:
                result_set = pickle.load(f)
        else:
            results = []
            for seed in range(0, 10000, int(10000 / args.n)):
                util.seeder(seed)

                config = GPT2Config.from_pretrained(
                    "gpt2", output_hidden_states=True)
                model_args = Args(
                    args = ("gpt2",),
                    kwargs = {"output_hidden_states": True}
                )
                model = GPT2Model(config)
                model = model.eval()

                tokenizer_args = Args(
                    args=("gpt2",)
                )
                tokenizer = GPT2TokenizerFast.from_pretrained(*tokenizer_args.args)
                tokenizer.add_tokens(TOKENS[granularity])
                model.resize_token_embeddings(len(tokenizer))

                feature_extractor = PassageTokenizer(
                    tokenizer, sentence_preprocessor=PREPROCESSORS[granularity])

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", scipy.stats.PearsonRConstantInputWarning)  # Ignore the very many PearsonRCoefficient warnings
                    raw_scores, scores = score(model, feature_extractor, seed=seed)

                result = PereiraResult(
                    seed=seed,
                    scores=scores,
                    raw_scores=raw_scores,
                    model_args=model_args,
                    tokenizer_args=tokenizer_args
                )
                results.append(result)

            result_set = PereiraResultSet(results=results)
            with open(gpt2_pos_result, "wb") as f:
                pickle.dump(result_set, f)

        for result in result_set.results:
            scores = result.scores
            normalized_scores = normalize_scores(scores)
            print(
                f"Granularity: {granularity}",
                f"Seed: {result.seed}",
                f"Scores {scores}",
                f"Normalized scores: {normalized_scores}"
            )

if __name__ == "__main__":
    main()
