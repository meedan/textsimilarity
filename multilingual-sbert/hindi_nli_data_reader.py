import pandas as pd
from sentence_transformers.readers import InputExample


class HindiNLIDataReader(object):
    """
    Reads in the XNLI dataset
    """
    def get_examples(self):
        bhaav = pd.read_csv("../data/recasted-hindi-nli-data/bhaav/bhaav_recasted.tsv", sep="\t")
        bhaav = bhaav.dropna(subset=['entailment'])
        mr = pd.read_csv("../data/recasted-hindi-nli-data/MR/recasted_movie_review_data.tsv", sep="\t")
        pr = pd.read_csv("../data/recasted-hindi-nli-data/PR/recasted_product_review_data.tsv", sep="\t")

        examples = []
        idx = 0
        for _, item in pd.concat([bhaav, mr, pr]).iterrows():
            guid = idx
            idx += 1
            sentence1 = item['context']
            sentence2 = item['hypothesis']
            label = item['entailment']
            examples.append(InputExample(guid=guid, texts=[sentence1, sentence2], label=self.map_label(label)))

        return examples

    @staticmethod
    def get_labels():
        return {"not-entailed": 0, "not entailed": 0, "entailed": 1}

    def get_num_labels(self):
        return 2

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]
