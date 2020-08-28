

from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import csv
import os
from zipfile import ZipFile
import random

cur_dir = os.path.split(os.path.realpath(__file__))[0]
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


# As base model, we use DistilBERT-base that was pre-trained on NLI and STSb data
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Training for multiple epochs can be beneficial, as in each epoch a mini-batch is sampled differently
# hence, we get different negatives for each positive
num_epochs = 10

# Increasing the batch size improves the performance for MultipleNegativesRankingLoss. Choose it as large as possible
# I achieved the good results with a batch size of 300-350 (requires about 30 GB of GPU memory)
train_batch_size = 64

dataset_path = os.path.join(cur_dir, 'quora-IR-dataset')
model_save_path = os.path.join(cur_dir,'output/training_MultipleNegativesRankingLoss-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

os.makedirs(model_save_path, exist_ok=True)

# Check if the dataset exists. If not, download and extract
if not os.path.exists(dataset_path):
    logging.info("Dataset not found. Download")
    zip_save_path = os.path.join(cur_dir, 'quora-IR-dataset.zip')
    util.http_get(url='https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/quora-IR-dataset.zip', path=zip_save_path)
    with ZipFile(zip_save_path, 'r') as zip:
        zip.extractall(dataset_path)


######### Read train data  ##########
train_samples = []
with open(os.path.join(dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['is_duplicate'] == '1':
            train_samples.append(InputExample(texts=[row['question1'], row['question2']], label=1))
            train_samples.append(InputExample(texts=[row['question2'], row['question1']], label=1))  #if A is a duplicate of B, then B is a duplicate of A


# After reading the train_samples, we create a SentencesDataset and a DataLoader
train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)


###### Duplicate Questions Information Retrieval ######
evaluators = []
# Given a question and a large corpus of thousands questions, find the most relevant (i.e. duplicate) question
# in that corpus.

# For faster processing, we limit the development corpus to only 10,000 sentences.
max_corpus_size = 100000

ir_queries = {}             #Our queries (qid => question)
ir_needed_qids = set()      #QIDs we need in the corpus
ir_corpus = {}              #Our corpus (qid => question)
ir_relevant_docs = {}       #Mapping of relevant documents for a given query (qid => set([relevant_question_ids])

with open(os.path.join(dataset_path, 'information-retrieval/dev-queries.tsv'), encoding='utf8') as fIn:
    next(fIn)   # Skip header
    for line in fIn:
        qid, query, duplicate_ids = line.strip().split('\t')
        duplicate_ids = duplicate_ids.split(',')
        ir_queries[qid] = query
        ir_relevant_docs[qid] = set(duplicate_ids)

        for qid in duplicate_ids:
            ir_needed_qids.add(qid)

# First get all needed relevant documents (i.e., we must ensure, that the relevant questions are actually in the corpus
distraction_questions = {}
with open(os.path.join(dataset_path, 'information-retrieval/corpus.tsv'), encoding='utf8') as fIn:
    next(fIn)   # Skip header
    for line in fIn:
        qid, question = line.strip().split('\t')

        if qid in ir_needed_qids:
            ir_corpus[qid] = question
        else:
            distraction_questions[qid] = question

# Now, also add some irrelevant questions to fill our corpus
other_qid_list = list(distraction_questions.keys())
random.shuffle(other_qid_list)

for qid in other_qid_list[0:max(0, max_corpus_size-len(ir_corpus))]:
    ir_corpus[qid] = distraction_questions[qid]

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