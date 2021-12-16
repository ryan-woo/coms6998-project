import os
import pickle
from pathlib import Path
from typing import Callable

import torch
from torch.nn.functional import pad

from surfboard.sound import Waveform
from google.cloud import texttospeech

# TODO:
#   - sound based character tokenization (PIE vocab)
#   - vocabularies of different sizes (no language bias)
#       - 26 + 26^2 + 26^3 + 26^4


class FeatureExtractor:

    def extract_input_features(self, stimulus_ids, sentences):
        raise Exception("Not implemented!")


class SentenceTokenizer(FeatureExtractor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # https://github.com/huggingface/transformers/issues/8452
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.config.pad_token_id = self.config.eos_token_id

    def extract_input_features(self, stimulus_ids, sentences):
        assert len(stimulus_ids) == len(sentences)

        tokenized = self.tokenizer(
            sentences,
            padding=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        # note that the ending character here is usually a period
        # (we can experiment w/ the last word by subtracting 1)
        output_coords = []
        for idx, input_ids in enumerate(tokenized['input_ids']):
            input_ids = input_ids.tolist()
            if tk.pad_token_id in input_ids:
                output_coords.append((idx, input_ids.index(tk.pad_token_id) - 1))
            else:
                output_coords.append((idx, len(input_ids) - 1))

        return tokenized, output_coords


class PassageTokenizer(FeatureExtractor):
    def __init__(self, tokenizer, sentence_delimiter: str = " ", sentence_preprocessor: Callable = None):
        self.tokenizer = tokenizer
        self.sentence_delimiter = sentence_delimiter
        self.sentence_preprocessor = sentence_preprocessor

    def extract_input_features(self, stimulus_ids, sentences):
        assert len(stimulus_ids) == len(sentences)

        if self.sentence_preprocessor is not None:
            sentences = [self.sentence_preprocessor(sentence) for sentence in sentences]

        stimulus_ends = []
        length_so_far = 0
        for sentence in sentences:
            length_so_far += len(sentence)
            stimulus_ends.append(length_so_far - 1)

            # we'll join the sentences with spaces
            length_so_far += 1

        tokenized = self.tokenizer(
            [self.sentence_delimiter.join(sentences)],
            add_special_tokens=True,
            return_tensors='pt'
        )

        # note that the ending character here is usually a period
        # (we can experiment w/ the last word by subtracting 1)
        output_coords = [
            (0, tokenized.char_to_token(stimulus_end)) for stimulus_end in stimulus_ends
        ]

        return tokenized, output_coords


PREPROCESSING_DIR = Path(__file__).parent.resolve() / 'preprocessed'
MP3_DIR = os.path.join(PREPROCESSING_DIR, 'mp3')

NUM_MFCCS = 36
class MFCCExtractor(FeatureExtractor):
    def __init__(self):
        self.mfcc_filepath = os.path.join(PREPROCESSING_DIR, 'mfccs.pkl')
        if os.path.exists(self.mfcc_filepath):
            with open(self.mfcc_filepath, 'rb') as f:
                self.mfccs = pickle.load(f)
        else:
            self.mfccs = {}


    def get_mfcc(self, sentence_id, sentence):
        if sentence_id not in self.mfccs:
            mp3_filepath = self.get_mp3_file(sentence_id, sentence)
            self.mfccs[sentence_id] = self.generate_mfccs(mp3_filepath)
            with open(self.mfcc_filepath, 'wb') as f:
                pickle.dump(self.mfccs, f)

        return self.mfccs[sentence_id]


    def get_mp3_file(self, sentence_id, sentence):
        mp3_filepath = os.path.join(MP3_DIR, "%s-male.mp3" % sentence_id)

        if os.path.exists(mp3_filepath):
            return mp3_filepath

        print("generating mp3 for %s..." % sentence_id)

        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=sentence)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-D",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )

        with open(mp3_filepath, "wb") as out:
            out.write(response.audio_content)

        return mp3_filepath

    def generate_mfccs(self, mp3_filepath):
        sound = Waveform(mp3_filepath)
        return torch.tensor(sound.mfcc(n_mfcc=NUM_MFCCS)).transpose(1, 0)


class MFCCSentenceExtractor(MFCCExtractor):
    def extract_input_features(self, stimulus_ids, sentences):
        assert len(stimulus_ids) == len(sentences)
        
        stimuli = []
        largest_sequence = 0
        for stimulus_id, sentence in zip(stimulus_ids, sentences):
            stimulus = self.get_mfcc(stimulus_id, sentence)
            largest_sequence = max(largest_sequence, stimulus.shape[0])
            stimuli.append(stimulus)

        model_inputs = {
            'inputs_embeds': torch.stack([
                pad(stimulus, ((0, largest_sequence - stimulus.shape[0]), (0, 0)))
                for stimulus in stimuli
            ]),
            'attention_mask': torch.stack([
                pad(torch.ones(stimulus.shape[0]), ((0, largest_sequence - stimulus.shape[0])))
                for stimulus in stimuli
            ])
        }

        output_coords = [
            (idx, stimulus.shape[0]) for idx, stimulus in enumerate(stimuli)
        ]

        return model_inputs, output_coords


class MFCCPassageExtractor(MFCCExtractor):
    def extract_input_features(self, stimulus_ids, sentences):
        assert len(stimulus_ids) == len(sentences)
        
        stimuli = []
        stimulus_ends = []
        last_stimulus_end = 0
        for stimulus_id, sentence in zip(stimulus_ids, sentences):
            stimulus = self.get_mfcc(stimulus_id, sentence)
            last_stimulus_end += stimulus.shape[0]
            stimuli.append(stimulus)
            stimulus_ends.append(last_stimulus_end - 1)

        model_inputs = {
            'inputs_embeds': torch.cat(stimuli).unsqueeze(dim=0),
            'attention_mask': torch.ones((1, last_stimulus_end))
        }

        output_coords = [
            (0, stimulus_end) for stimulus_end in stimulus_ends
        ]

        return model_inputs, output_coords

        

