import csv
import json
import random
from enum import Enum

import cld3
import numpy as np
import pandas as pd
from utils import remove_emoji, spam_list, contains_url, contains_phone_number, get_sbert_embedding, get_fuzzy_similarity_score, vcosine
from elasticsearch import Elasticsearch
from elasticsearch import helpers

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
    data.drop(data[(data['message_text'].map(contains_phone_number)) | (data['message_text'].map(contains_url)) |
                   (data['message_text'].map(len) > 1200) | (data['message_text'].map(len) < 60)].index, inplace=True)
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
                         tip['text'] != 'NA' and not tip['text'].isspace() and 'language' in tip and (
                                     60 <= len(tip['text']) <= 1200) and not contains_url(
                             tip['text']) and not contains_phone_number(tip['text'])]

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


def sample_with_elasticsearch():
    es = Elasticsearch()
    group_sample_size = 1000
    tipline_data = load_tip_line_claim_data('../data/tiplines.csv')
    public_groups_data = load_public_group_data('../data/pg_text_spamfree.csv')
    factcheck_data = load_factcheck_data('../data/claim_reviews_with_language.json')
    languages = ['en', 'pt', 'hi', 'hi-Latn', 'mr', 'bn', 'ta', 'te', 'ml']
    [es.indices.delete(index=language.lower(), ignore=[400, 404]) for language in languages]
    samples = []
    for language in languages:
        print('Starting processing for language={}'.format(language))
        language_tip_line_data = [item for item in tipline_data if item['language'] == language]
        language_factcheck_data = [item for item in factcheck_data if item['language'] == language]
        language_public_groups_data = [item for item in public_groups_data if item['language'] == language]

        # populating ES
        es.indices.create(index=language.lower(), ignore=400)
        actions = [
            {
                "_index": language.lower(),
                "_id": i,
                "_source": item
            }
            for i, item in enumerate(language_tip_line_data + language_factcheck_data + language_public_groups_data)
        ]
        helpers.bulk(es, actions)

        print('Populated DB for language={}'.format(language))

        # sampling
        sample_carry = 2  # 1 because 500 isn't divisble by 3 and we take 2 more sample out of public groups data
        language_sample_size = group_sample_size // 6 if language != 'pt' else group_sample_size // 4
        tipline_sample = random.sample(language_tip_line_data, min(language_sample_size, len(language_tip_line_data)))
        if len(tipline_sample) < language_sample_size:
            sample_carry += language_sample_size - len(tipline_sample)
        factcheck_sample = random.sample(language_factcheck_data, min(language_sample_size, len(language_factcheck_data)))
        if len(factcheck_sample) < language_sample_size:
            sample_carry += language_sample_size - len(factcheck_sample)
        public_groups_sample = random.sample(language_public_groups_data, min(group_sample_size // 3 + sample_carry, len(language_public_groups_data)))
        for item in tipline_sample + factcheck_sample + public_groups_sample:
            # query elasticsearch
            query_body = {
                "query": {
                    "match": {
                        "text": item['text']
                    }
                }
            }
            results = es.search(index=language.lower(), body=query_body)
            result_texts = [result_item['_source']['text'] for result_item in results['hits']['hits']]

            if len(result_texts) == 0:
                continue

            results_embeddings = get_sbert_embedding(result_texts, language if language != 'hi-Latn' else 'hi')
            sample_embedding = get_sbert_embedding(item['text'], language if language != 'hi-Latn' else 'hi')
            cosine_distances = vcosine(sample_embedding, results_embeddings)

            # capturing matches that aren't duplicates
            best_match_idx = None
            for i, distance in enumerate(cosine_distances[0]):
                fuzzy_score = get_fuzzy_similarity_score(item['text'], result_texts[i])
                if fuzzy_score < 0.9 and distance >= 0.8:
                    best_match_idx = i
                    samples.append(results['hits']['hits'][i]['_source'])
                    break

            no_best_match_found = best_match_idx is None
            best_match_idx = 0 if no_best_match_found else best_match_idx
            matches_70_80 = []
            matches_80_90 = []
            for i in reversed(range(best_match_idx+1, len(result_texts))):
                if len(matches_70_80) >= 2 and len(matches_80_90) >= 2:
                    break
                if cosine_distances[0][i] < 0.7 or cosine_distances[0][i] >= 0.9:
                    continue
                if len(matches_70_80) < 2 and 0.7 <= cosine_distances[0][i] < 0.8:
                    fuzzy_score = get_fuzzy_similarity_score(item['text'], result_texts[i])
                    if fuzzy_score < 0.9:
                        matches_70_80.append(results['hits']['hits'][i]['_source'])
                elif len(matches_80_90) < 2 and 0.8 <= cosine_distances[0][i] < 0.9:
                    fuzzy_score = get_fuzzy_similarity_score(item['text'], result_texts[i])
                    if fuzzy_score < 0.9:
                        matches_80_90.append(results['hits']['hits'][i]['_source'])

            if len(matches_70_80) > 0 and len(matches_80_90) > 0:
                samples.append(matches_70_80[random.choice([0, 1]) if len(matches_70_80) > 1 else 0])
                samples.append(matches_80_90[random.choice([0, 1]) if len(matches_80_90) > 1 else 0])
            elif len(matches_70_80) > 0 and len(matches_80_90) == 0:
                samples += matches_70_80
            elif len(matches_80_90) > 0 and len(matches_70_80) == 0:
                samples += matches_80_90

            if not (no_best_match_found and len(matches_70_80) == 0 and len(matches_80_90) == 0):
                samples.append(item)

    temp_samples = []
    for language in languages:
        language_samples = [sample for sample in samples if sample['language'] == language]
        language_samples = random.sample(language_samples, min(group_sample_size, len(language_samples)))
        temp_samples += language_samples

    samples = temp_samples
    with open('textsimilarity_samples_es.json', 'w') as fp:
        json.dump(samples, fp)

if __name__ == '__main__':
    # sample_data_from_all_sources()
    sample_with_elasticsearch()
