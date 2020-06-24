from sentence_transformers.readers import InputExample
import json


class ClaimPairDataReader(object):
    """
    Reads in the XNLI dataset
    """
    def get_examples(self, language, split='train'):
        pairs = self._load_data()
        if language == 'hi':
            pairs = pairs['hindi_headlines']
        elif language == 'pt':
            pairs = pairs['ciper']
        else:
            pairs = pairs['fact_pairs']
        split_point = int(round(len(pairs)*0.8))
        if split == 'train':
            pairs = pairs[:split_point]
        else:
            pairs = pairs[split_point:]
        examples = []
        for i, item in enumerate(pairs):
            guid = i
            sentence1 = item['lookup_text']
            sentence2 = item['database_text']
            label = item['label']
            examples.append(InputExample(guid=guid, texts=[sentence1, sentence2], label=label))

        return examples

    @staticmethod
    def get_labels():
        return [0, 1]

    def get_num_labels(self):
        return len(self.get_labels())

    def _load_data(self):
        with open('../data/multilingual_sentence_matched_datasets.json') as f:
            sentence_pairs = json.load(f)
        return sentence_pairs