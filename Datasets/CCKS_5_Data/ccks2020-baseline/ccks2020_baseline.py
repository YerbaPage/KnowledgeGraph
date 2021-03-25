#!/usr/bin/env python
# coding: utf-8

# # CCKS 2020: 基于本体的金融知识图谱自动化构建技术评测
# 
# 竞赛背景
# 金融研报是各类金融研究结构对宏观经济、金融、行业、产业链以及公司的研究报告。报告通常是有专业人员撰写，对宏观、行业和公司的数据信息搜集全面、研究深入，质量高，内容可靠。报告内容往往包含产业、经济、金融、政策、社会等多领域的数据与知识，是构建行业知识图谱非常关键的数据来源。另一方面，由于研报本身所容纳的数据与知识涉及面广泛，专业知识众多，不同的研究结构和专业认识对相同的内容的表达方式也会略有差异。这些特点导致了从研报自动化构建知识图谱困难重重，解决这些问题则能够极大促进自动化构建知识图谱方面的技术进步。
#  
# 本评测任务参考 TAC KBP 中的 Cold Start 评测任务的方案，围绕金融研报知识图谱的自动化图谱构建所展开。评测从预定义图谱模式（Schema）和少量的种子知识图谱开始，从非结构化的文本数据中构建知识图谱。其中图谱模式包括 10 种实体类型，如机构、产品、业务、风险等；19 个实体间的关系，如(机构，生产销售，产品)、(机构，投资，机构)等；以及若干实体类型带有属性，如（机构，英文名）、（研报，评级）等。在给定图谱模式和种子知识图谱的条件下，评测内容为自动地从研报文本中抽取出符合图谱模式的实体、关系和属性值，实现金融知识图谱的自动化构建。所构建的图谱在大金融行业、监管部门、政府、行业研究机构和行业公司等应用非常广泛，如风险监测、智能投研、智能监管、智能风控等，具有巨大的学术价值和产业价值。
#  
# 评测本身不限制各参赛队伍使用的模型、算法和技术。希望各参赛队伍发挥聪明才智，构建各类无监督、弱监督、远程监督、半监督等系统，迭代的实现知识图谱的自动化构建，共同促进知识图谱技术的进步。
# 
# 竞赛任务
# 本评测任务参考 TAC KBP 中的 Cold Start 评测任务的方案，围绕金融研报知识图谱的自动化图谱构建所展开。评测从预定义图谱模式（Schema）和少量的种子知识图谱开始，从非结构化的文本数据中构建知识图谱。评测本身不限制各参赛队伍使用的模型、算法和技术。希望各参赛队伍发挥聪明才智，构建各类无监督、弱监督、远程监督、半监督等系统，迭代的实现知识图谱的自动化构建，共同促进知识图谱技术的进步。
# 
# 主办方邮箱  wangwenguang@datagrand.com kdd.wang@gmail.com
# 
# 
# 参考：https://www.biendata.com/competition/ccks_2020_5/

# In[41]:


import json
import logging
import os
import random
import re
import base64
from collections import defaultdict
from pathlib import Path

# import attr
import tqdm
import hanlp
import numpy as np
import torch
import torch.optim
import torch.utils.data
from torch.nn import functional as F
from torchcrf import CRF
from pytorch_transformers import BertModel, BertTokenizer
import jieba
from jieba.analyse.tfidf import TFIDF
from jieba.posseg import POSTokenizer
import jieba.posseg as pseg
from itertools import product
from IPython.display import HTML


# In[ ]:





# # 预处理函数
# 
# 对文章进行预处理，切分句子和子句等

# In[2]:


def split_to_sents(content, filter_length=(2, 1000)):
    content = re.sub(r"\s*", "", content)
    content = re.sub("([。！…？?!；;])", "\\1\1", content)
    sents = content.split("\1")
    sents = [_[: filter_length[1]] for _ in sents]
    return [_ for _ in sents
            if filter_length[0] <= len(_) <= filter_length[1]]

def split_to_subsents(content, filter_length=(2, 1000)):
    content = re.sub(r"\s*", "", content)
    content = re.sub("([。！…？?!；;,，])", "\\1\1", content)
    sents = content.split("\1")
    sents = [_[: filter_length[1]] for _ in sents]
    return [_ for _ in sents
            if filter_length[0] <= len(_) <= filter_length[1]]


# In[3]:


def read_json(file_path):
    with open(file_path, mode='r', encoding='utf8') as f:
        return json.load(f)


# # 预训练模型配置
# 
# 参考 https://github.com/huggingface/pytorch-transformers 下载预训练模型，并配置下面参数为相关路径
# 
# ```python
# PRETRAINED_BERT_MODEL_DIR = '/you/path/to/bert-base-chinese/' 
# ```

# In[4]:


PRETRAINED_BERT_MODEL_DIR = './model/' 


# # 一些参数

# In[5]:


DATA_DIR = './data'  # 输入数据文件夹
OUT_DIR = './output'  # 输出文件夹

Path(OUT_DIR).mkdir(exist_ok=True)

BATCH_SIZE = 32
TOTAL_EPOCH_NUMS = 10
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
YANBAO_DIR_PATH = str(Path(DATA_DIR, 'yanbao_txt'))
SAVE_MODEL_DIR = str(OUT_DIR)
print(DEVICE, flush=True)


# ## 读入原始数据
# 
# - 读入：所有研报内容
# - 读入：原始训练实体数据

# In[6]:


yanbao_texts = []
for yanbao_file_path in Path(YANBAO_DIR_PATH).glob('*.txt'):
    with open(yanbao_file_path) as f:
        yanbao_texts.append(f.read())
#         if len(yanbao_texts) == 10:
#             break

# 来做官方的实体训练集，后续会混合来自第三方工具，规则，训练数据来扩充模型训练数据
to_be_trained_entities = read_json(Path(DATA_DIR, 'entities.json'))


# # 用hanlp进行实体识别
# 
# hanlp支持对人物、机构的实体识别，可以使用它来对其中的两个实体类型进行识别：人物、机构。
# 
# hanlp见[https://github.com/hankcs/HanLP](https://github.com/hankcs/HanLP)

# In[7]:


## NER by third party tool
class HanlpNER:
    def __init__(self):
        self.recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
        self.max_sent_len = 126
        self.ent_type_map = {
            'NR': '人物',
            'NT': '机构'
        }
        self.black_list = {'公司'}

    def recognize(self, sent):
        entities_dict = {}
        for result in self.recognizer.predict([list(sent)]):
            for entity, hanlp_ent_type, _, _ in result:
                if not re.findall(r'^[\.\s\da-zA-Z]{1,2}$', entity) and                         len(entity) > 1 and entity not in self.black_list                         and hanlp_ent_type in self.ent_type_map:
                    entities_dict.setdefault(self.ent_type_map[hanlp_ent_type], []).append(entity)
        return entities_dict


# In[8]:


def aug_entities_by_third_party_tool():
    hanlpner = HanlpNER()
    entities_by_third_party_tool = defaultdict(list)
    for file in tqdm.tqdm(list(Path(DATA_DIR, 'yanbao_txt').glob('*.txt'))[:]):
        with open(file, encoding='utf-8') as f:
            sents = [[]]
            cur_sent_len = 0
            for line in f:
                for sent in split_to_subsents(line):
                    sent = sent[:hanlpner.max_sent_len]
                    if cur_sent_len + len(sent) > hanlpner.max_sent_len:
                        sents.append([sent])
                        cur_sent_len = len(sent)
                    else:
                        sents[-1].append(sent)
                        cur_sent_len += len(sent)
            sents = [''.join(_) for _ in sents]
            sents = [_ for _ in sents if _]
            for sent in sents:
                entities_dict = hanlpner.recognize(sent)
                for ent_type, ents in entities_dict.items():
                    entities_by_third_party_tool[ent_type] += ents

    for ent_type, ents in entities_by_third_party_tool.items():
        entities_by_third_party_tool[ent_type] = list([ent for ent in set(ents) if len(ent) > 1])
    return entities_by_third_party_tool


# In[10]:


# 此任务十分慢, 但是只需要运行一次
entities_by_third_party_tool = aug_entities_by_third_party_tool()
for ent_type, ents in entities_by_third_party_tool.items():
    to_be_trained_entities[ent_type] = list(set(to_be_trained_entities[ent_type] + ents))


# In[11]:


for k, v in entities_by_third_party_tool.items():
    print(k)
    print(set(v))


# ## 通过规则抽取实体
# 
# - 机构
# - 研报
# - 文章
# - 风险

# In[12]:


def aug_entities_by_rules(yanbao_dir):
    entities_by_rule = defaultdict(list)
    for file in list(yanbao_dir.glob('*.txt'))[:]:
        with open(file, encoding='utf-8') as f:
            found_yanbao = False
            found_fengxian = False
            for lidx, line in enumerate(f):
                # 公司的标题
                ret = re.findall('^[\(（]*[\d一二三四五六七八九十①②③④⑤]*[\)）\.\s]*(.*有限公司)$', line)
                if ret:
                    entities_by_rule['机构'].append(ret[0])
                    
                # 研报
                if not found_yanbao and lidx <= 5 and len(line) > 10:
                    may_be_yanbao = line.strip()
                    if not re.findall(r'\d{4}\s*[年-]\s*\d{1,2}\s*[月-]\s*\d{1,2}\s*日?', may_be_yanbao)                             and not re.findall('^[\d一二三四五六七八九十]+\s*[\.、]\s*.*$', may_be_yanbao)                             and not re.findall('[\(（]\d+\.*[A-Z]*[\)）]', may_be_yanbao)                             and len(may_be_yanbao) > 5:
                        entities_by_rule['研报'].append(may_be_yanbao)
                        found_yanbao = True

                # 文章
                for sent in split_to_sents(line):
                    results = re.findall('《(.*?)》', sent)
                    for result in results:
                        entities_by_rule['文章'].append(result)     

                # 风险
                for sent in split_to_sents(line):
                    if found_fengxian:
                        sent = sent.split('：')[0]
                        fengxian_entities = re.split('以及|、|，|；|。', sent)
                        fengxian_entities = [re.sub('^[■]+[\d一二三四五六七八九十①②③④⑤]+', '', ent) for ent in fengxian_entities]
                        fengxian_entities = [re.sub('^[\(（]*[\d一二三四五六七八九十①②③④⑤]+[\)）\.\s]+', '', ent) for ent in fengxian_entities]
                        fengxian_entities = [_ for _ in fengxian_entities if len(_) >=4]
                        entities_by_rule['风险'] += fengxian_entities
                        found_fengxian = False
                    if not found_fengxian and re.findall('^\s*[\d一二三四五六七八九十]*\s*[\.、]*\s*风险提示[:：]*$', sent):
                        found_fengxian = True
                    
                    results = re.findall('^\s*[\d一二三四五六七八九十]*\s*[\.、]*\s*风险提示[:：]*(.{5,})$', sent)
                    if results:
                        fengxian_entities = re.split('以及|、|，|；|。', results[0])
                        fengxian_entities = [re.sub('^[■]+[\d一二三四五六七八九十①②③④⑤]+', '', ent) for ent in fengxian_entities]
                        fengxian_entities = [re.sub('^[\(（]*[\d一二三四五六七八九十①②③④⑤]+[\)）\.\s]+', '', ent) for ent in fengxian_entities]
                        fengxian_entities = [_ for _ in fengxian_entities if len(_) >=4]
                        entities_by_rule['风险'] += fengxian_entities
                    
    for ent_type, ents in entities_by_rule.items():
        entities_by_rule[ent_type] = list(set(ents))
    return entities_by_rule


# In[13]:


# 通过规则来寻找新的实体
entities_by_rule = aug_entities_by_rules(Path(DATA_DIR, 'yanbao_txt'))
for ent_type, ents in entities_by_rule.items():
    to_be_trained_entities[ent_type] = list(set(to_be_trained_entities[ent_type] + ents))


# In[14]:


for k, v in entities_by_rule.items():
    print(k)
    print(set(v))


# # 定义NER模型
# 

# In[15]:


class BertCRF(torch.nn.Module):
    def __init__(self, pretrained_bert_model_file_path, num_tags: int, batch_first: bool = False, hidden_size=768):
        super(BertCRF, self).__init__()
        self.bert_module = BertModel.from_pretrained(pretrained_bert_model_file_path)
        self.tag_linear = torch.nn.Linear(hidden_size, num_tags)
        self.crf_module = CRF(num_tags, batch_first)

    def forward(self,
                inputs_ids,
                tags,
                mask = None,
                token_type_ids=None,
                reduction = 'mean'
                ) -> torch.Tensor:
        bert_outputs = self.bert_module.forward(inputs_ids, attention_mask=mask, token_type_ids=token_type_ids)[0]
        bert_outputs = F.dropout(bert_outputs, p=0.2, training=self.training)
        bert_outputs = self.tag_linear(bert_outputs)
        score = -self.crf_module.forward(bert_outputs, tags=tags, mask=mask, reduction=reduction)
        return score

    def decode(self,
               input_ids,
               attention_mask=None,
               token_type_ids=None
               ):
        bert_outputs = self.bert_module.forward(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )[0]
        bert_outputs = self.tag_linear(bert_outputs)
        best_tags_list = self.crf_module.decode(bert_outputs, mask=attention_mask)
        return best_tags_list


# ## 定义NER标签

# In[16]:


chinese_entity_type_vs_english_entity_type = {
    '人物': 'People',
    '行业': 'Industry',
    '业务': 'Business',
    '研报': 'Report',
    '机构': 'Organization',
    '风险': 'Risk',
    '文章': 'Article',
    '指标': 'Indicator',
    '品牌': 'Brand',
    '产品': 'Product'
}
english_entity_type_vs_chinese_entity_type = {v: k for k, v in chinese_entity_type_vs_english_entity_type.items()}

START_TAG = "[CLS]"
END_TAG = "[SEP]"
O = "O"

BPeople = "B-People"
IPeople = "I-People"
BIndustry = "B-Industry"
IIndustry = "I-Industry"
BBusiness = 'B-Business'
IBusiness = 'I-Business'
BProduct = 'B-Product'
IProduct = 'I-Product'
BReport = 'B-Report'
IReport = 'I-Report'
BOrganization = 'B-Organization'
IOrganization = 'I-Organization'
BRisk = 'B-Risk'
IRisk = 'I-Risk'
BArticle = 'B-Article'
IArticle = 'I-Article'
BIndicator = 'B-Indicator'
IIndicator = 'I-Indicator'
BBrand = 'B-Brand'
IBrand = 'I-Brand'

PAD = "[PAD]"
UNK = "[UNK]"
tag2idx = {
    START_TAG: 0,
    END_TAG: 1,
    O: 2,
    BPeople: 3,
    IPeople: 4,
    BIndustry: 5,
    IIndustry: 6,
    BBusiness: 7,
    IBusiness: 8,
    BProduct: 9,
    IProduct: 10,
    BReport: 11,
    IReport: 12,
    BOrganization: 13,
    IOrganization: 14,
    BRisk: 15,
    IRisk: 16,
    BArticle: 17,
    IArticle: 18,
    BIndicator: 19,
    IIndicator: 20,
    BBrand: 21,
    IBrand: 22,
}
tag2id = tag2idx
idx2tag = {v: k for k, v in tag2idx.items()}


# ## 预处理数据函数
# 
# `preprocess_data` 函数中的 `for_train` 参数比较重要，指示是否是训练集
# 
# 由于给定的训练数据实体部分没有给定出现的位置，这里需要自行查找到实体出现的位置
# 
# - 如果是训练集, 按照`entities_json`中的内容在文章中寻找位置并标注, 并将训练数据处理成bio形式
# - 测试数据仅仅做了分句并转化成token id

# In[17]:


class Article:
    def __init__(self, text):
        self._text = text
        self.para_texts = self.split_into_paras(self._text)
        self.sent_texts = [self.split_into_sentence(para) for para in self.para_texts]

    def fix_text(self, text: str) -> str:
        paras = text.split('\n')
        paras = list(filter(lambda para: len(para.strip()) != 0, paras))
        return '\n'.join(paras)

    def split_into_paras(self, text: str):
        paras = list(filter(lambda para: len(para.strip()) != 0, text.split('\n')))
        return paras

    def split_into_sentence(self, one_para_text: str, splited_puncs = None):
        if splited_puncs is None:
            splited_puncs = ['。', '？', '！']
        splited_re_pattern = '[' + ''.join(splited_puncs) + ']'

        para = one_para_text
        sentences = re.split(splited_re_pattern, para)
        sentences = list(filter(lambda sent: len(sent) != 0, sentences))

        return sentences

    def find_sents_by_entity_name(self, entity_text):
        ret_sents = []
        if entity_text not in self._text:
            return []
        else:
            for para in self.split_into_paras(self._text):
                if entity_text not in para:
                    continue
                else:
                    for sent in self.split_into_sentence(para):
                        if entity_text in sent:
                            ret_sents.append(sent)
        return ret_sents


# In[18]:


def _find_all_start_end(source, target):
    if not target:
        return []
    occurs = []
    offset = 0
    while offset < len(source):
        found = source[offset:].find(target)
        if found == -1:
            break
        else:
            occurs.append([offset + found, offset + found + len(target) - 1])
        offset += (found + len(target))
    return occurs

def preprocess_data(entities_json,
                    article_texts,
                    tokenizer: BertTokenizer,
                    for_train: bool = True):
    """
    [{
        'sent': xxx, 'entity_name': yyy, 'entity_type': zzz, 'start_token_id': 0, 'end_token_id': 5,
        'start_index': 0, 'end_index': 2, 
            'sent_tokens': ['token1', 'token2'], 'entity_tokens': ['token3', 'token4']
    }]
    """

    preprocessed_datas = []

    all_sents = []
    for article in tqdm.tqdm([Article(t) for t in article_texts]):
        for para_text in article.para_texts:
            for sent in article.split_into_sentence(para_text):
                sent_tokens = list(sent)
                entity_labels = []
                for entity_type, entities in entities_json.items():
                    for entity_name in entities:
                        if entity_name not in sent:
                            continue
                        all_sents.append(sent)
                        start_end_indexes = _find_all_start_end(sent, entity_name)
                        assert len(start_end_indexes) >= 1
                        for str_start_index, str_end_index in start_end_indexes:
                            entity_tokens = list(entity_name)

                            one_entity_label = {
                                'entity_type': entity_type,
                                'start_token_id': str_start_index,
                                'end_token_id': str_end_index,
                                'start_index': str_start_index,
                                'end_index': str_end_index,
                                'entity_tokens': entity_tokens,
                                'entity_name': entity_name
                            }
                            entity_labels.append(one_entity_label)

                if not entity_labels:
                    tags = [O for _ in range(len(sent_tokens))]
                    tag_ids = [tag2idx[O] for _ in range(len(sent_tokens))]
                    if for_train:
                        continue
                else:
                    tags = []
                    tag_ids = []
                    for sent_token_index in range(len(sent_tokens)):
                        tag = O
                        for entity_label in entity_labels:
                            if sent_token_index == entity_label['start_token_id']:
                                tag = f'B-{chinese_entity_type_vs_english_entity_type[entity_label["entity_type"]]}'
                            elif entity_label['start_token_id'] < sent_token_index < entity_label["end_token_id"]:
                                tag = f'I-{chinese_entity_type_vs_english_entity_type[entity_label["entity_type"]]}'
                        tag_id = tag2idx[tag]
                        tags.append(tag)
                        tag_ids.append(tag_id)
                assert len(sent_tokens) == len(tags) == len(tag_ids)
                not_o_indexes = [index for index, tag in enumerate(tags) if tag != O]
                all_entities = [sent_tokens[index] for index in not_o_indexes]
                all_entities2 = entity_labels

                preprocessed_datas.append({
                    'sent': sent,
                    'sent_tokens': sent_tokens,
                    'sent_token_ids': tokenizer.convert_tokens_to_ids(sent_tokens),
                    'entity_labels': entity_labels,
                    'tags': tags,
                    'tag_ids': tag_ids
                })
    return preprocessed_datas


# # 定义dataset 以及 dataloader

# In[19]:


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_datas, tokenizer: BertTokenizer, max_length=512 ):
        self.preprocessed_datas = preprocessed_datas
        self.tokenizer = tokenizer
        self.max_length = max_length

    def pad_sent_ids(self, sent_ids, max_length, padded_token_id):
        mask = [1] * (min(len(sent_ids), max_length)) + [0] * (max_length - len(sent_ids))
        sent_ids = sent_ids[:max_length] + [padded_token_id] * (max_length - len(sent_ids))
        return sent_ids, mask

    def process_one_preprocessed_data(self, preprocessed_data):
        import copy
        preprocessed_data = copy.deepcopy(preprocessed_data)
        
        sent_token_ids = self.tokenizer.convert_tokens_to_ids(
            [START_TAG]) + preprocessed_data['sent_token_ids'] + self.tokenizer.convert_tokens_to_ids([END_TAG])

        sent_token_ids, mask = self.pad_sent_ids(
            sent_token_ids, max_length=self.max_length, padded_token_id=self.tokenizer.pad_token_id)

        sent_token_ids = np.array(sent_token_ids)
        mask = np.array(mask)
        
        preprocessed_data['sent'] = '^' + preprocessed_data['sent'] + '$'
        preprocessed_data['sent_tokens'] = [START_TAG] + preprocessed_data['sent_tokens'] + [END_TAG]
        preprocessed_data['sent_token_ids'] = sent_token_ids
        
        
        tags = [START_TAG] + preprocessed_data['tags'] + [END_TAG]
        tag_ids = [tag2idx[START_TAG]] + preprocessed_data['tag_ids'] + [tag2idx[END_TAG]]
        tag_ids, _ = self.pad_sent_ids(tag_ids, max_length=self.max_length, padded_token_id=tag2idx[O])
        tag_ids = np.array(tag_ids)
        

        for entity_label in preprocessed_data['entity_labels']:
            entity_label['start_token_id'] += 1
            entity_label['end_token_id'] += 1
            entity_label['start_index'] += 1
            entity_label['end_index'] += 1
        
        
        preprocessed_data['tags'] = tags
        preprocessed_data['tag_ids'] = tag_ids

        not_o_indexes = [index for index, tag in enumerate(preprocessed_data['tags']) if tag != O]

        not_o_indexes_str = not_o_indexes
        all_entities = [preprocessed_data['sent_tokens'][index] for index in not_o_indexes]
        all_entities2 = preprocessed_data['entity_labels']
        all_entities3 = [preprocessed_data['sent'][index] for index in not_o_indexes_str]
        
        preprocessed_data.update({'mask': mask})

        return preprocessed_data

    def __getitem__(self, item):
        return self.process_one_preprocessed_data(
            self.preprocessed_datas[item]
        )

    def __len__(self):
        return len(self.preprocessed_datas)


def custom_collate_fn(data):
    # copy from torch official，无需深究
    from torch._six import container_abcs, string_classes

    r"""Converts each NumPy array data field into a tensor"""
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        tmp_dict = {}
        for key in data:
            if key in ['sent_token_ids', 'tag_ids', 'mask']:
                tmp_dict[key] = custom_collate_fn(data[key])
                if key == 'mask':
                    tmp_dict[key] = tmp_dict[key].byte()
            else:
                tmp_dict[key] = data[key]
        return tmp_dict
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(custom_collate_fn(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [custom_collate_fn(d) for d in data]
    else:
        return data


def build_dataloader(preprocessed_datas, tokenizer: BertTokenizer, batch_size=32, shuffle=True):
    dataset = MyDataset(preprocessed_datas, tokenizer)
    import torch.utils.data
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=shuffle)
    return dataloader


# # 定义训练时评价指标
# 
# 仅供训练时参考, 包含实体的precision，recall以及f1。
# 
# 只有和标注的数据完全相同才算是1，否则为0

# In[20]:


# 训练时指标
class EvaluateScores:
    def __init__(self, entities_json, predict_entities_json):
        self.entities_json = entities_json
        self.predict_entities_json = predict_entities_json

    def compute_entities_score(self):
        return evaluate_entities(self.entities_json, self.predict_entities_json, list(set(self.entities_json.keys())))
    
def _compute_metrics(ytrue, ypred):
    ytrue = set(ytrue)
    ypred = set(ypred)
    tr = len(ytrue)
    pr = len(ypred)
    hit = len(ypred.intersection(ytrue))
    p = hit / pr if pr!=0 else 0
    r = hit / tr if tr!=0 else 0
    f1 = 2 * p * r / (p + r) if (p+r)!=0 else 0
    return {
        'p': p,
        'r': r,
        'f': f1,
    }


def evaluate_entities(true_entities, pred_entities, entity_types):
    scores = []

    ps2 = []
    rs2 = []
    fs2 = []
    
    for ent_type in entity_types:

        true_entities_list = true_entities.get(ent_type, [])
        pred_entities_list = pred_entities.get(ent_type, [])
        s = _compute_metrics(true_entities_list, pred_entities_list)
        scores.append(s)
    ps = [i['p'] for i in scores]
    rs = [i['r'] for i in scores]
    fs = [i['f'] for i in scores]
    s = {
        'p': sum(ps) / len(ps),
        'r': sum(rs) / len(rs),
        'f': sum(fs) / len(fs),
    }
    return s


# ## 定义ner train loop， evaluate loop ，test loop

# In[21]:


def train(model: BertCRF, optimizer, data_loader: torch.utils.data.DataLoader, logger: logging.Logger, epoch_id,
          device='cpu'):
    pbar = tqdm.tqdm(data_loader)
    for batch_id, one_data in enumerate(pbar):
        model.train()

        sent_token_ids = torch.stack([d['sent_token_ids'] for d in one_data]).to(device)
        tag_ids = torch.stack([d['tag_ids'] for d in one_data]).to(device)
        mask = torch.stack([d['mask'] for d in one_data]).to(device)

        loss = model.forward(sent_token_ids, tag_ids, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description('epoch: {}, loss: {:.3f}'.format(epoch_id, loss.item()))


def evaluate(
        model, data_loader: torch.utils.data.DataLoader, logger: logging.Logger,
        tokenizer, device='cpu',
):
    founded_entities_json = defaultdict(set)
    golden_entities_json = defaultdict(set)
    for batch_id, one_data in enumerate(data_loader):
        model.eval()
        sent_token_ids = torch.stack([d['sent_token_ids'] for d in one_data]).to(device)
        tag_ids = torch.stack([d['tag_ids'] for d in one_data]).to(device)
        mask = torch.stack([d['mask'] for d in one_data]).to(device)

        best_tag_ids_list = model.decode(sent_token_ids, attention_mask=mask)
        best_tags_list = [[idx2tag[idx] for idx in idxs] for idxs in best_tag_ids_list]

        for data, best_tags in zip(one_data, best_tag_ids_list):

            for entity_label in data['entity_labels']:
                golden_entities_json[entity_label['entity_type']].add(entity_label['entity_name'])

            record = False
            for token_index, tag_id in enumerate(best_tags):
                tag = idx2tag[tag_id]
                if tag.startswith('B'):
                    start_token_index = token_index
                    entity_type = tag[2:]
                    record = True
                elif record and tag == O:
                    end_token_index = token_index

                    str_start_index = start_token_index
                    str_end_index = end_token_index

                    entity_name = data['sent'][str_start_index: str_end_index]

                    entity_type = english_entity_type_vs_chinese_entity_type[entity_type]
                    founded_entities_json[entity_type].add(entity_name)
                    record = False
    evaluate_tool = EvaluateScores(golden_entities_json, founded_entities_json)
    scores = evaluate_tool.compute_entities_score()
    return scores['f']


def test(model, data_loader: torch.utils.data.DataLoader, logger: logging.Logger, device):
    founded_entities = []
    for batch_id, one_data in enumerate(tqdm.tqdm(data_loader)):
        model.eval()
        sent_token_ids = torch.stack([d['sent_token_ids'] for d in one_data]).to(device)
        mask = torch.stack([d['mask'] for d in one_data]).to(device)

        with torch.no_grad():
            best_tag_ids_list = model.decode(sent_token_ids, attention_mask=mask, token_type_ids=None)

        for data, best_tags in zip(one_data, best_tag_ids_list):
            record = False
            for token_index, tag_id in enumerate(best_tags):
                tag = idx2tag[tag_id]
                if tag.startswith('B'):
                    start_token_index = token_index
                    entity_type = tag[2:]
                    record = True
                elif record and tag == O:
                    end_token_index = token_index
                    entity_name = data['sent_tokens'][start_token_index: end_token_index + 1]
                    founded_entities.append((entity_name, entity_type, data['sent']))
                    record = False
    result = defaultdict(list)
    for entity_name, entity_type, sent in founded_entities:
        entity = ''.join(entity_name).replace('##', '')
        entity = entity.replace('[CLS]', '')
        entity = entity.replace('[UNK]', '')
        entity = entity.replace('[SEP]', '')
        if len(entity) > 1:
            result[english_entity_type_vs_chinese_entity_type[entity_type]].append((entity, sent))

    for ent_type, ents in result.items():
        result[ent_type] = list(set(ents))
    return result


# # ner主要训练流程
# 
# - 分隔训练集验证集，并处理成dataset dataloader
# - 训练，验证，保存模型

# In[22]:


def main_train(logger, tokenizer, model, to_be_trained_entities, yanbao_texts):
    entities_json = to_be_trained_entities
    train_entities_json = {k: [] for k in entities_json}
    dev_entities_json = {k: [] for k in entities_json}

    train_proportion = 0.9
    for entity_type, entities in entities_json.items():
        entities = entities.copy()
        random.shuffle(entities)
        
        train_entities_json[entity_type] = entities[: int(len(entities) * train_proportion)]
        dev_entities_json[entity_type] = entities[int(len(entities) * train_proportion):]

    
    train_preprocessed_datas = preprocess_data(train_entities_json, yanbao_texts, tokenizer)
    train_dataloader = build_dataloader(train_preprocessed_datas, tokenizer, batch_size=BATCH_SIZE)
    
    dev_preprocessed_datas = preprocess_data(dev_entities_json, yanbao_texts, tokenizer)
    dev_dataloader = build_dataloader(dev_preprocessed_datas, tokenizer, batch_size=BATCH_SIZE)

    model = model.to(DEVICE)
    for name, param in model.named_parameters():
        if "bert_module" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    optimizer = torch.optim.Adam([para for para in model.parameters() if para.requires_grad],
                                 lr=0.001,
                                 weight_decay=0.0005)
    best_evaluate_score = 0
    for epoch in range(TOTAL_EPOCH_NUMS):
        train(model, optimizer, train_dataloader, logger=logger, epoch_id=epoch, device=DEVICE)
        evaluate_score = evaluate(model, dev_dataloader, logger=logger, tokenizer=tokenizer, device=DEVICE)
        print('评估分数：', evaluate_score)
        if evaluate_score >= best_evaluate_score:
            best_evaluate_score = evaluate_score
            save_model_path = os.path.join(SAVE_MODEL_DIR, 'finnal_ccks_model.pth')
            logger.info('saving model to {}'.format(save_model_path))
            torch.save(model.cpu().state_dict(), save_model_path)
            model.to(DEVICE)


# ## 准备训练ner模型

# In[47]:


logger = logging.getLogger(__name__)

tokenizer = BertTokenizer.from_pretrained(
    os.path.join(PRETRAINED_BERT_MODEL_DIR, 'vocab.txt'))

model = BertCRF(pretrained_bert_model_file_path=PRETRAINED_BERT_MODEL_DIR,
                num_tags=len(tag2id),
                batch_first=True)

save_model_path = os.path.join(SAVE_MODEL_DIR, 'finnal_ccks_model.pth')
if Path(save_model_path).exists():
    model_state_dict = torch.load(save_model_path, map_location='cpu')
    model.load_state_dict(model_state_dict)


# In[46]:





# In[48]:


# 训练数据在main_train函数中处理并生成dataset dataloader，此处无需生成

# 测试数据在此处处理并生成dataset dataloader
test_preprocessed_datas = preprocess_data({},
                                          yanbao_texts,
                                          tokenizer,
                                          for_train=False)
test_dataloader = build_dataloader(test_preprocessed_datas,
                                   tokenizer,
                                   batch_size=BATCH_SIZE)


# ## 整个训练流程是：
# 
# - 使用数据集增强得到更多的实体
# - 使用增强过后的实体来指导训练
# 
# 
# - 训练后的模型重新对所有文档中进行预测，得到新的实体，加入到实体数据集中
# - 使用扩增后的实体数据集来进行二次训练，再得到新的实体，再增强实体数据集
# - (模型预测出来的数据需要`review_model_predict_entities`后处理形成提交格式)
# 
# 
# - 如果提交结果，需要`extract_entities`函数删除提交数据中那些出现在训练数据中的实体

# ### 模型预测结果后处理函数
# 
# - `review_model_predict_entities`函数将模型预测结果后处理，从而生成提交文件格式

# In[49]:


def review_model_predict_entities(model_predict_entities):
    word_tag_map = POSTokenizer().word_tag_tab
    idf_freq = TFIDF().idf_freq
    reviewed_entities = defaultdict(list)
    for ent_type, ent_and_sent_list in model_predict_entities.items():
        for ent, sent in ent_and_sent_list:
            start = sent.lower().find(ent)
            if start == -1:
                continue
            start += 1
            end = start + len(ent) - 1
            tokens = jieba.lcut(sent)
            offset = 0
            selected_tokens = []
            for token in tokens:
                offset += len(token)
                if offset >= start:
                    selected_tokens.append(token)
                if offset >= end:
                    break

            fixed_entity = ''.join(selected_tokens)
            fixed_entity = re.sub(r'\d*\.?\d+%$', '', fixed_entity)
            if ent_type == '人物':
                if len(fixed_entity) >= 10:
                    continue
            if len(fixed_entity) <= 1:
                continue
            if re.findall(r'^\d+$', fixed_entity):
                continue
            if word_tag_map.get(fixed_entity,
                                '') == 'v' and idf_freq[fixed_entity] < 7:
                continue
            reviewed_entities[ent_type].append(fixed_entity)
    return reviewed_entities


# - `extract_entities` 删除与训练集中重复的实体

# In[50]:


def extract_entities(to_be_trained_entities):
    test_entities = to_be_trained_entities
    train_entities = read_json(Path(DATA_DIR, 'entities.json'))

    for ent_type, ents in test_entities.items():
        test_entities[ent_type] = list(
            set(ents) - set(train_entities[ent_type]))

    for ent_type in train_entities.keys():
        if ent_type not in test_entities:
            test_entities[ent_type] = []
    return test_entities


# In[ ]:


# 循环轮次数目
nums_round = 1
for i in range(nums_round):
    # train
    main_train(logger, tokenizer, model, to_be_trained_entities, yanbao_texts)        
    
    model = model.to(DEVICE)
    model_predict_entities = test(model, test_dataloader, logger=logger, device=DEVICE)
    
    # 修复训练预测结果
    reviewed_entities = review_model_predict_entities(model_predict_entities)
    
    # 将训练预测结果再次放入训练集中， 重新训练或者直接出结果
    for ent_type, ents in reviewed_entities.items():
        to_be_trained_entities[ent_type] = list(set(to_be_trained_entities[ent_type] + ents))

# 创造出提交结果
submit_entities = extract_entities(to_be_trained_entities)


# # 属性抽取
# 
# 通过规则抽取属性
# 
# - 研报时间
# - 研报评级
# - 文章时间

# In[ ]:


def find_article_time(yanbao_txt, entity):
    str_start_index = yanbao_txt.index(entity)
    str_end_index = str_start_index + len(entity)
    para_start_index = yanbao_txt.rindex('\n', 0, str_start_index)
    para_end_index = yanbao_txt.index('\n', str_end_index)

    para = yanbao_txt[para_start_index + 1: para_end_index].strip()
    if len(entity) > 5:
        ret = re.findall(r'(\d{4})\s*[年-]\s*(\d{1,2})\s*[月-]\s*(\d{1,2})\s*日?', para)
        if ret:
            year, month, day = ret[0]
            time = '{}/{}/{}'.format(year, month.lstrip(), day.lstrip())
            return time

    start_index = 0
    time = None
    min_gap = float('inf')
    for word, poseg in pseg.cut(para):
        if poseg in ['t', 'TIME'] and str_start_index <= start_index < str_end_index:
            gap = abs(start_index - (str_start_index + str_end_index) // 2)
            if gap < min_gap:
                min_gap = gap
                time = word
        start_index += len(word)
    return time


def find_yanbao_time(yanbao_txt, entity):
    paras = [para.strip() for para in yanbao_txt.split('\n') if para.strip()][:5]
    for para in paras:
        ret = re.findall(r'(\d{4})\s*[\./年-]\s*(\d{1,2})\s*[\./月-]\s*(\d{1,2})\s*日?', para)
        if ret:
            year, month, day = ret[0]
            time = '{}/{}/{}'.format(year, month.lstrip(), day.lstrip())
            return time
    return None


# In[ ]:


def extract_attrs(entities_json):
    train_attrs = read_json(Path(DATA_DIR, 'attrs.json'))['attrs']

    seen_pingjis = []
    for attr in train_attrs:
        if attr[1] == '评级':
            seen_pingjis.append(attr[2])
    article_entities = entities_json.get('文章', [])
    yanbao_entities = entities_json.get('研报', [])

    attrs_json = []
    for file_path in tqdm.tqdm(list(Path(DATA_DIR, 'yanbao_txt').glob('*.txt'))):
        yanbao_txt = '\n' + Path(file_path).open().read() + '\n'
        for entity in article_entities:
            if entity not in yanbao_txt:
                continue
            time = find_article_time(yanbao_txt, entity)
            if time:
                attrs_json.append([entity, '发布时间', time])

        yanbao_txt = '\n'.join(
            [para.strip() for para in yanbao_txt.split('\n') if
             len(para.strip()) != 0])
        for entity in yanbao_entities:
            if entity not in yanbao_txt:
                continue

            paras = yanbao_txt.split('\n')
            for para_id, para in enumerate(paras):
                if entity in para:
                    break

            paras = paras[: para_id + 5]
            for para in paras:
                for pingji in seen_pingjis:
                    if pingji in para:
                        if '上次' in para:
                            attrs_json.append([entity, '上次评级', pingji])
                            continue
                        elif '维持' in para:
                            attrs_json.append([entity, '上次评级', pingji])
                        attrs_json.append([entity, '评级', pingji])

            time = find_yanbao_time(yanbao_txt, entity)
            if time:
                attrs_json.append([entity, '发布时间', time])
    attrs_json = list(set(tuple(_) for _ in attrs_json) - set(tuple(_) for _ in train_attrs))
    
    return attrs_json


# In[ ]:


train_attrs = read_json(Path(DATA_DIR, 'attrs.json'))['attrs']
submit_attrs = extract_attrs(submit_entities)


# In[ ]:


submit_attrs


# # 关系抽取
# 
# - 对于研报实体，整个文档抽取特定类型(行业，机构，指标)的关系实体
# - 其他的实体仅考虑与其出现在同一句话中的其他实体组织成特定关系

# In[ ]:


def extract_relations(schema, entities_json):
    relation_by_rules = []
    relation_schema = schema['relationships']
    unique_s_o_types = []
    so_type_cnt = defaultdict(int)
    for s_type, p, o_type in schema['relationships']:
        so_type_cnt[(s_type, o_type)] += 1
    for (s_type, o_type), cnt in so_type_cnt.items():
        if cnt == 1 and s_type != o_type:
            unique_s_o_types.append((s_type, o_type))

    for path in tqdm.tqdm(list(Path(DATA_DIR, 'yanbao_txt').glob('*.txt'))):
        with open(path) as f:
            entity_dict_in_file = defaultdict(lambda: defaultdict(list))
            main_org = None
            for line_idx, line in enumerate(f.readlines()):
                for sent_idx, sent in enumerate(split_to_sents(line)):
                    for ent_type, ents in entities_json.items():
                        for ent in ents:
                            if ent in sent:
                                if ent_type == '机构' and len(line) - len(ent) < 3 or                                         re.findall('[\(（]\d+\.*[A-Z]*[\)）]', line):
                                    main_org = ent
                                else:
                                    if main_org and '客户' in sent:
                                        relation_by_rules.append([ent, '客户', main_org])
                                entity_dict_in_file[ent_type][
                                    ('test', ent)].append(
                                    [line_idx, sent_idx, sent,
                                     sent.find(ent)]
                                )

            for s_type, p, o_type in relation_schema:
                s_ents = entity_dict_in_file[s_type]
                o_ents = entity_dict_in_file[o_type]
                if o_type == '业务' and not '业务' in line:
                    continue
                if o_type == '行业' and not '行业' in line:
                    continue
                if o_type == '文章' and not ('《' in line or not '》' in line):
                    continue
                if s_ents and o_ents:
                    for (s_ent_src, s_ent), (o_ent_src, o_ent) in product(s_ents, o_ents):
                        if s_ent != o_ent:
                            s_occs = [tuple(_[:2]) for _ in
                                      s_ents[(s_ent_src, s_ent)]]
                            o_occs = [tuple(_[:2]) for _ in
                                      o_ents[(o_ent_src, o_ent)]]
                            intersection = set(s_occs) & set(o_occs)
                            if s_type == '研报' and s_ent_src == 'test':
                                relation_by_rules.append([s_ent, p, o_ent])
                                continue
                            if not intersection:
                                continue
                            if (s_type, o_type) in unique_s_o_types and s_ent_src == 'test':
                                relation_by_rules.append([s_ent, p, o_ent])

    train_relations = read_json(Path(DATA_DIR, 'relationships.json'))['relationships']
    result_relations_set = list(set(tuple(_) for _ in relation_by_rules) - set(tuple(_) for _ in train_relations))
    return result_relations_set


# In[ ]:


schema = read_json(Path(DATA_DIR, 'schema.json'))
submit_relations  = extract_relations(schema, submit_entities)


# In[ ]:


submit_relations


# ## 生成提交文件
# 
# 根据biendata的要求生成提交文件
# 
# 参考：https://www.biendata.com/competition/ccks_2020_5/make-submission/

# In[ ]:


final_answer = {'attrs': submit_attrs,
                'entities': submit_entities,
                'relationships': submit_relations,
                }


with open('output/answers.json', mode='w') as fw:
        json.dump(final_answer, fw, ensure_ascii=False, indent=4)


# In[ ]:


with open('output/answers.json', 'rb') as fb:
    data = fb.read()

b64 = base64.b64encode(data)
payload = b64.decode()
html = '<a download="{filename}" href="data:text/json;base64,{payload}" target="_blank">{title}</a>'
html = html.format(payload=payload,title='answers.json',filename='answers.json')
HTML(html)

