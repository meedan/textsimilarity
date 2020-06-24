from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, readers, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from datetime import datetime

import random
import json
import csv
import logging
import sys
import torch
import os
import numpy as np
def cosine_similarity(vecA, vecB):
  """
  Return the cosine simularity between two vectors vecA and vecB.
  @return float between 0 and 1 inclusive. One indicates identical.
  """
  csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
  if np.isnan(np.sum(csim)):
    return 0
  return csim

#We can pass multiple train files to this script
train_files = sys.argv[1:]
"""
!!!here I use a train_file of:
!!!http://devingaffney.com/files/opus_en-hi.txt
!!!which is derived from:
opus_read --directory JW300 \
    --source en \
    --target hi \
    --write en-hi.txt\
    --write_mode moses
(https://pypi.org/project/opustools-pkg/ to use opus_read)
"""
def generate_evaluation_file():
  headlines = json.loads(open("data/hindi_headlines.json").read())
  datasets = []
  for dataset in [headlines[:-60], headlines[-60:]]:
    tmp_dataset = []
    for row in dataset:
      random_row = random.sample(headlines, 1)[0]
      while random_row["database_text"] == row["database_text"]:
        random_row = random.sample(headlines, 1)[0]
      tmp_dataset.append([row["lookup_text"], row["database_text"], 4])
      tmp_dataset.append([row["lookup_text"], random_row["database_text"], 0])
    datasets.append(tmp_dataset)
  train, test = datasets
  with open('data/hindi_sbert_sts_train.csv', 'w', newline='') as f:
      writer = csv.writer(f, delimiter="\t")
      writer.writerows(train)
  with open('data/hindi_sbert_sts_test.csv', 'w', newline='') as f:
      writer = csv.writer(f, delimiter="\t")
      writer.writerows(test)

generate_evaluation_file()
if len(train_files) == 0:
    print("Please specify at least 1 training file: python training_multilingual.py path/to/trainfile.txt")

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

max_seq_length = 128
train_batch_size = 64

logging.info("Load teacher model")
teacher_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

logging.info("Create student model from scratch")
word_embedding_model = models.Transformer("xlm-roberta-base")

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

output_path = "output/make-multilingual-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.info("Create dataset reader")


###### Read Dataset ######
train_data = ParallelSentencesDataset(student_model=model, teacher_model=teacher_model)
for train_file in train_files:
    train_data.load_data(train_file)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=model)

# The below all apply to the de example - how does one evaluate the model outside this single example???
###### Load dev sets ######

# Test on STS 2017.en-de dataset using Spearman rank correlation
logging.info("Read data/hindi_sbert_sts_train.csv dataset")
evaluators = []
sts_reader = readers.STSDataReader('./data/', s1_col_idx=0, s2_col_idx=1, score_col_idx=2)
dev_data = SentencesDataset(examples=sts_reader.get_examples('hindi_sbert_sts_train.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator_sts = evaluation.EmbeddingSimilarityEvaluator(dev_dataloader, name='Hindi_Headlines_en_hi_sbert')
evaluators.append(evaluator_sts)

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1]),
          epochs=20,
          evaluation_steps=1000,
          warmup_steps=10000,
          scheduler='warmupconstant',
          output_path=output_path,
          save_best_model=True,
          optimizer_params= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
          )

headlines = json.loads(open("data/hindi_headlines.json").read())
results = [["Lookup Text", "Database Text", "Cosine Similarity", "Is Supposed to be Match"]]
for row in headlines:
  hit_vectors = model.encode([row["lookup_text"], row["database_text"]])
  random_row = random.sample(headlines, 1)[0]
  while random_row["database_text"] == row["database_text"]:
    random_row = random.sample(headlines, 1)[0]
  miss_vectors = model.encode([row["lookup_text"], random_row["database_text"]])
  results.append([row["lookup_text"], row["database_text"], cosine_similarity(hit_vectors[0], hit_vectors[1]), 1])
  results.append([row["lookup_text"], random_row["database_text"], cosine_similarity(miss_vectors[0], miss_vectors[1]), 0])

with open('data/hindi_sbert_output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(results)