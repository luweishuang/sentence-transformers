# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from sentence_transformers import losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, evaluation
import logging
from datetime import datetime
from pocess_data_zh import get_data, get_iq_corpus
import os
import random

cur_dir = os.path.split(os.path.realpath(__file__))[0]


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


# 作者在issues里提到的多语言的预训练模型 xlm-r-40langs-bert-base-nli-stsb-mean-tokens
# 针对信息检索任务的多语言预训练模型  distilbert-multilingual-nli-stsb-quora-ranking
model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')

# Training for multiple epochs can be beneficial, as in each epoch a mini-batch is sampled differently
# hence, we get different negatives for each positive
num_epochs = 10

# Increasing the batch size improves the performance for MultipleNegativesRankingLoss. Choose it as large as possible
# I achieved the good results with a batch size of 300-350 (requires about 30 GB of GPU memory)
train_batch_size = 64

model_save_path = os.path.join(cur_dir, 'output/training_MultipleNegativesRankingLoss-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

os.makedirs(model_save_path, exist_ok=True)

data_file = os.path.join(cur_dir, "data/LCCC-large.json")
train_samples = get_data(data_file)

# After reading the train_samples, we create a SentencesDataset and a DataLoader
train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)


###### Duplicate Questions Information Retrieval ######
evaluators = []
data_file = os.path.join(cur_dir,  "data/STC.json")
max_ir_num = 5000
max_corpus_size = 100000
ir_queries, ir_corpus, ir_relevant_docs = get_iq_corpus(data_file, max_ir_num, max_corpus_size)

# Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR
# metrices. For our use case MRR@k and Accuracy@k are relevant.
ir_evaluator = evaluation.InformationRetrievalEvaluator(ir_queries, ir_corpus, ir_relevant_docs)
evaluators.append(ir_evaluator)

# Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
# We optimize the model with respect to the score from the last evaluator (scores[-1])
seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

logging.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=seq_evaluator,
          epochs=num_epochs,
          warmup_steps=1000,
          output_path=model_save_path,
          output_path_ignore_not_empty=True
          )