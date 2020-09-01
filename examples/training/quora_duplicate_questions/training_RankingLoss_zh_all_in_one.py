# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, evaluation
import logging
from datetime import datetime
import os
import re
import json
import cn2an      # 阿拉伯数字 <=> 中文数字
import string
import numpy as np
import unicodedata
from zipfile import ZipFile
from sentence_transformers.readers import InputExample


cur_dir = os.path.split(os.path.realpath(__file__))[0]

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


all_flag = string.punctuation + u'“《》「」』『·—□〈〉•’●‘×”・∫,?!.♪:⦆⦆╮╭〜😂👏💨✨◤◢☀€😍🙀ノ♥★⋯⋯σ≪≫♡⎢◊.|:—.↓∩'
pattern_flag = re.compile('[%s]' % re.escape(all_flag))
alpha_char = string.punctuation + u'abcdefghijklmnopqrstuvwxyz'
pattern_alpha = re.compile('[%s]' % re.escape(alpha_char))

trantab = str.maketrans('，。！？【】（）〔〕％＃＠＆１２３４５６７８９０、', ',.!?[]()[]%#@&1234567890,')


def is_all_chinese1(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def filter_emoji(desstr,restr=''):
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)


def data_clean(str_in):
    text_a = str_in.strip().replace(" ", "").replace("alink", "").replace("°C", "度")
    text_b = filter_emoji(text_a, restr='')
    text_1 = unicodedata.normalize('NFKC', text_b.lower().replace(" ", ""))      # 中文标点转换为英文标点
    text_2 = text_1.translate(trantab)  # 漏网之鱼手动修改对应
    text_3 = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", text_2)   # 去掉小括号(), {}, []里的内容
    text_4 = cn2an.transform(text_3, "an2cn")                   # 阿拉伯数字转中文
    text_5 = pattern_flag.sub(u'', text_4)                      # 去掉标点符号
    if not is_all_chinese1(text_5):
        text_6 = pattern_alpha.sub(u'', text_5)
        if not is_all_chinese1(text_6):
            return ""
    return text_3


def get_data(data_file):
    train_samples = []
    discard_num = 0
    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
        for cur_dialg in dataset:
            query_sent = data_clean(cur_dialg[0])
            content_sent = data_clean(cur_dialg[1])
            if len(query_sent) == 0 or len(content_sent) == 0:
                discard_num += 1
                continue
            else:
                train_samples.append(InputExample(texts=[query_sent, content_sent], label=1))   # 无论单轮还是多轮，只取最相关的前两句
    return train_samples


def get_iq_corpus(data_file, max_ir_num, max_corpus_size):
    ir_queries = {}             #Our queries (qid => question)
    ir_corpus = {}              #Our corpus (qid => question)
    ir_relevant_docs = {}       #Mapping of relevant documents for a given query (qid => set([relevant_question_ids])
    # 验证集需要获取的变量有 ir_queries(ir_queries[qid] = query)5000, ir_corpus(ir_corpus[qid] = question)100000, ir_relevant_docs(ir_relevant_docs[qid] = set(duplicate_ids))
    discard_num = 0
    use_num = 0
    index_array = np.random.shuffle(np.arange(max_corpus_size))
    with open(data_file, "r", encoding="utf-8") as fr:
        dataset = json.loads(fr.read())
        for k, v in dataset.items():
            for cur_dialg in v:
                query_sent = data_clean(cur_dialg[0])
                content_sent = data_clean(cur_dialg[1])
                if len(query_sent) == 0 or len(content_sent) == 0:
                    discard_num += 1
                    continue
                else:
                    if use_num + 1 < max_corpus_size:
                        if use_num < max_ir_num:
                            ir_queries[index_array[use_num]] = query_sent
                            ir_relevant_docs[index_array[use_num]] = set([index_array[use_num + 1]])
                        ir_corpus[index_array[use_num]] = query_sent
                        ir_corpus[index_array[use_num+1]] = content_sent
                    else:
                        break
                    use_num += 2
    return ir_queries, ir_corpus, ir_relevant_docs


def downloads_dataset(dataset_name):
    dataset_path = os.path.join(cur_dir, 'data', dataset_name)
    if not os.path.exists(dataset_path):
        logging.info("Dataset not found. Download")
        zip_save_path = os.path.join(cur_dir, 'LCCC-large.zip')
        util.http_get(
            url='https://coai-dataset.oss-cn-beijing.aliyuncs.com/%s.zip' % dataset_name,
            path=zip_save_path)
        with ZipFile(zip_save_path, 'r') as zip:
            zip.extractall(dataset_path)


def cli_main():
    # 作者在issues里提到的多语言的预训练模型 xlm-r-40langs-bert-base-nli-stsb-mean-tokens
    # 针对信息检索任务的多语言预训练模型  distilbert-multilingual-nli-stsb-quora-ranking
    model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')

    num_epochs = 10
    train_batch_size = 64
    model_save_path = os.path.join(cur_dir, 'output/training_MultipleNegativesRankingLoss-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(model_save_path, exist_ok=True)

    downloads_dataset("LCCC-large")
    downloads_dataset("STC-corpus")

    data_file = os.path.join(cur_dir, "data/LCCC-large/LCCD.json")
    train_samples = get_data(data_file)

    # After reading the train_samples, we create a SentencesDataset and a DataLoader
    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    ###### Duplicate Questions Information Retrieval ######
    evaluators = []
    data_file = os.path.join(cur_dir, "data/STC-corpus/STC.json")
    max_ir_num = 5000
    max_corpus_size = 100000
    ir_queries, ir_corpus, ir_relevant_docs = get_iq_corpus(data_file, max_ir_num, max_corpus_size)

    ir_evaluator = evaluation.InformationRetrievalEvaluator(ir_queries, ir_corpus, ir_relevant_docs)
    evaluators.append(ir_evaluator)
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


if __name__ == "__main__":
    cli_main()