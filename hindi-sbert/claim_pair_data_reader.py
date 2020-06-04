from sentence_transformers.readers import InputExample
import json


class ClaimPairDataReader(object):
    """
    Reads in the XNLI dataset
    """
    def get_examples(self, language=None):
        hindi_pairs = self._load_data()

        examples = []
        for i, item in enumerate(hindi_pairs):
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
        return sentence_pairs['hindi_headlines']