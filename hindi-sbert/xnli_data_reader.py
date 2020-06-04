from sentence_transformers.readers import InputExample
import json


class XNLIDataReader(object):
    """
    Reads in the XNLI dataset
    """
    def get_examples(self, language):
        with open('data/xnli/xnli.dev.jsonl', 'r') as json_file:
            json_list = list(json_file)
            xnli_data = [json.loads(line) for line in json_list]

        with open('data/xnli/xnli.test.jsonl', 'r') as json_file:
            json_list = list(json_file)
            xnli_data += [json.loads(line) for line in json_list]

        xnli_data = [item for item in xnli_data if item['language'] == language]

        examples = []
        for item in xnli_data:
            guid = item['pairID']
            sentence1 = item['sentence1']
            sentence2 = item['sentence2']
            label = item['gold_label']
            examples.append(InputExample(guid=guid, texts=[sentence1, sentence2], label=self.map_label(label)))

        return examples

    @staticmethod
    def get_labels():
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]
