import logging

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, models, readers, evaluation, \
    losses
from sentence_transformers.datasets import ParallelSentencesDataset
from torch.utils.data import DataLoader
from claim_pair_data_reader import ClaimPairDataReader

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

max_seq_length = 128
train_batch_size = 32

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
# logging.info("Loading previously trained student-teacher model")
# model = SentenceTransformer('models/hindi-sxlmr-stmodel')

output_path = 'models/se-asian-sbert'

logging.info("Create dataset reader")

###### Read Dataset ######
train_file_path = 'train_southeast_asian_parallel_corpus.txt'
train_data = ParallelSentencesDataset(student_model=model, teacher_model=teacher_model)
train_data.load_data(train_file_path)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=model)

###### Load dev sets ######

# Test on STS 2017.en-de dataset using Spearman rank correlation
logging.info("Read dev dataset")
evaluators = []
claim_pair_reader = ClaimPairDataReader()
dev_data = SentencesDataset(examples=claim_pair_reader.get_examples(split='train', language='hi'), model=model)
# dev_file_path = 'test_southeast_asian_parallel_corpus.txt'
# dev_data = ParallelSentencesDataset(student_model=model, teacher_model=teacher_model)
# dev_data.load_data(dev_file_path)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator_sts = evaluation.EmbeddingSimilarityEvaluator(dev_dataloader, name='SE Asian Test Data')
evaluators.append(evaluator_sts)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1]),
          epochs=2,
          evaluation_steps=1000,
          warmup_steps=10000,
          scheduler='warmupconstant',
          output_path=output_path,
          save_best_model=True,
          optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
          )
