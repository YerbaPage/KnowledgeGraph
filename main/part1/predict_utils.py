# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import pandas as pd
from tqdm import tqdm
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam
from apex import amp


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name: 
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=0.05, t_total=len(train_iter) * config.num_epochs)
    # optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=0.05)
                         
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model.eval()
 
    dev_acc, dev_loss = evaluate(config, model, dev_iter)
    print('\n Evaluating, Val Loss: {:.3f}  Val Acc: {:.3f}  '.format(dev_loss, dev_acc))

    predict(config, model, test_iter, dev_acc)

def predict(config, model, data_iter, dev_acc):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    logits_all = np.zeros(config.num_classes)
    print('\n ============== Predicting ==============')
    with torch.no_grad():
        pbar = tqdm(total=len(data_iter))
        for texts, labels in data_iter:
            outputs = model(texts)
            # loss = F.cross_entropy(outputs, labels)
            # loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            logits = outputs.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            logits_all = np.vstack([logits_all, logits])
            pbar.update(1)
        pbar.close()

    data = pd.DataFrame(predict_all)
    save_name = 'out/pred_{}_{:.3f}.csv'.format(config.model_name, dev_acc)
    data.to_csv(save_name)

    logits = pd.DataFrame(logits_all)
    save_name_logits = 'out/logits_{}_{:.3f}.csv'.format(config.model_name, dev_acc)
    logits.to_csv(save_name_logits, header=False)
    print('============== Predcition finifhed ==============')


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    logits_all = np.zeros(config.num_classes)
    with torch.no_grad():
        pbar = tqdm(total=len(data_iter))
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            logits = outputs.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            logits_all = np.vstack([logits_all, logits])
            pbar.update(1)
        pbar.close()
    acc = metrics.accuracy_score(labels_all, predict_all)
    logits = pd.DataFrame(logits_all)
    save_name_logits = 'out/test/test_logits_{}.csv'.format(config.model_name)
    logits.to_csv(save_name_logits, header=False)
    if test:
        report = metrics.classification_report(
            labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
