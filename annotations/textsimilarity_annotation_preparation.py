import csv
import json
import random
from enum import Enum

import cld3
import numpy as np
import pandas as pd
from cleaning import remove_emoji, spam_list, remove_urls, contains_url

random_seed = 72
random.seed(random_seed)


class SourceName(Enum):
    TIPLINE = 'tipline'
    PUBLICGROUPS = 'public_groups'
    FACTCHECK = 'fact-check'
    HEADLINES = 'headlines'


def is_spam(text, lang):
    return len([item for item in spam_list[lang] if item in text]) > 0


def load_public_group_data(path, group_sample_size=15000,
                           languages=['en', 'hi', 'hi-Latn', 'mr', 'bn', 'ta', 'te', 'ml']):
    data = pd.read_csv(path, delimiter='\t', header=0)
    data = data.loc[data['language'].isin(languages)]
    # data['message_text'] = data['message_text'].apply(lambda text: remove_urls(text))
    data.drop(data[data['message_text'].map(contains_url)].index, inplace=True)
    data_with_language = []
    for language in languages:
        df = data.loc[data['language'] == language]
        df = df.sample(min(group_sample_size, len(df)), random_state=random_seed)
        data_with_language += [{'text': item['message_text'], 'language': language, 'source': SourceName.PUBLICGROUPS.value} for i, item in df.iterrows()]

    return data_with_language


def load_tip_line_claim_data(path):
    with open(path) as csvfile:
        tip_line_requests = csv.reader(csvfile)
        tip_line_requests = [item for item in tip_line_requests]
    csv_headers = tip_line_requests[0]
    tip_line_requests = tip_line_requests[1:]

    temp_tip_line_requests = []
    for row in tip_line_requests:
        item = {}
        for i, key in enumerate(csv_headers):
            item[key] = row[i]
        temp_tip_line_requests.append(item)
    tip_line_requests = [item for item in temp_tip_line_requests if item['claim_type'] == 'Claim']
    return group_tiplines_by_language(tip_line_requests)


def load_factcheck_data(path):
    with open(path) as f:
        facts = json.load(f)
    facts = [item for item in facts if len(item['_source']['claim_review_headline']) > 20]
    return [{'text': item['_source']['claim_review_headline'], 'language': item['_source']['language'], 'source': SourceName.FACTCHECK.value} for item in facts]


def group_tiplines_by_language(tip_line_requests,
                               languages=['en', 'pt', 'hi', 'hi-Latn', 'mr', 'bn', 'ta', 'te', 'ml']):
    for tip in tip_line_requests:
        tip['text'] = remove_emoji(
            tip['media_text'] if tip['media_text'] != 'NA' and len(tip['media_text']) >= len(tip['media_title']) else
            tip['media_title'])
        lang_data = cld3.get_language(tip['text'])
        if lang_data is not None:
            tip['language'] = lang_data.language
    tip_line_requests = [tip for tip in tip_line_requests if
                         tip['text'] != 'NA' and not tip['text'].isspace() and 'language' in tip and len(
                             tip['text']) > 20 and not contains_url(tip['text'])]

    return [{'text': item['text'], 'language': item['language'], 'source': SourceName.TIPLINE.value}
            for item in tip_line_requests if item['language'] in languages]


def cosine_sim(vecA, vecB):
    """Find the cosine similarity distance between two vectors."""
    csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
    if np.isnan(np.sum(csim)):
        return 0
    return csim


def sample_data_from_all_sources():
    group_sample_size = 1000
    tipline_data = load_tip_line_claim_data('../data/tiplines.csv')
    public_groups_data = load_public_group_data('../data/pg_text_spamfree.csv')
    factcheck_data = load_factcheck_data('../data/claim_reviews_with_language.json')
    languages = ['en', 'pt', 'hi', 'hi-Latn', 'mr', 'bn', 'ta', 'te', 'ml']
    samples = []
    ids = np.random.choice(range(len(languages) * group_sample_size), size=len(languages) * group_sample_size,
                           replace=False).tolist()
    id_idx = 0
    for language in languages:
        sample_carry = 1 # 1 because 1000 isn't divisble by 3 and we take 1 more sample out of public groups data
        language_tip_line_data = [item for item in tipline_data if item['language'] == language]
        language_sample_size = group_sample_size // 3 if language != 'pt' else group_sample_size // 2
        tipline_sample = random.sample(language_tip_line_data, min(language_sample_size, len(language_tip_line_data)))
        if len(tipline_sample) < language_sample_size:
            sample_carry += language_sample_size - len(tipline_sample)
        language_factcheck_data = [item for item in factcheck_data if item['language'] == language]
        factcheck_sample = random.sample(language_factcheck_data, min(language_sample_size, len(language_factcheck_data)))
        if len(factcheck_sample) < language_sample_size:
            sample_carry += language_sample_size - len(factcheck_sample)
        language_public_groups_data = [item for item in public_groups_data if item['language'] == language]
        public_groups_sample = random.sample(language_public_groups_data, min(group_sample_size // 3 + sample_carry,
                                                                              len(language_public_groups_data)))
        for item in tipline_sample + factcheck_sample + public_groups_sample:
            item['id'] = ids[id_idx]
            id_idx += 1
            samples.append(item)

    with open('textsimilarity_samples.json', 'w') as fp:
        json.dump(samples, fp)


if __name__ == '__main__':
    sample_data_from_all_sources()
