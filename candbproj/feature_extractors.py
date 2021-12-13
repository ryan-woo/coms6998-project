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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_input_features(self, stimulus_ids, sentences):
        assert len(stimulus_ids) == len(sentences)
        
        stimulus_ends = []
        length_so_far = 0
        for sentence in sentences:
            length_so_far += len(sentence)
            stimulus_ends.append(length_so_far - 1)

            # we'll join the sentences with spaces
            length_so_far += 1

        tokenized = self.tokenizer(
            [' '.join(sentences)],
            add_special_tokens=True,
            return_tensors='pt'
        )

        # note that the ending character here is usually a period
        # (we can experiment w/ the last word by subtracting 1)
        output_coords = [
            (0, tokenized.char_to_token(stimulus_end)) for stimulus_end in stimulus_ends
        ]

        return tokenized, output_coords

        

