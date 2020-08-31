# -*- coding: utf-8 -*-

import re
import json
import cn2an      # 阿拉伯数字 <=> 中文数字
import string
import numpy as np
import unicodedata
from sentence_transformers.readers import InputExample


def get_data(data_file):
    train_samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
        for cur_dialg in dataset:
            train_samples.append(InputExample(texts=[cur_dialg[0].strip().replace(" ", ""), cur_dialg[1].strip().replace(" ", "")], label=1))   # 无论单轮还是多轮，只取最相关的前两句
    return train_samples


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


def data_clean1(str_in):
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
                query_sent = data_clean1(cur_dialg[0])
                content_sent = data_clean1(cur_dialg[1])
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


data_file = "data/STC.json"
max_ir_num = 5000
max_corpus_size = 100000
ir_queries, ir_corpus, ir_relevant_docs = get_iq_corpus(data_file, max_ir_num, max_corpus_size)