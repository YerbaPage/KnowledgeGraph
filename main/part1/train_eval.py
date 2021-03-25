# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import pandas as pd
from tqdm import tqdm
from utils import *
# from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained.optimization import BertAdam
from apex import amp
from torch.utils.tensorboard import SummaryWriter 
from datetime import datetime
# from focal_loss.focal_loss import *

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
    TIMESTAMP = "{}_bs{}_lr{}_ps{}_{:%Y_%m_%d_%H_%M_%S/}".format(config.model_name, config.batch_size, config.learning_rate, config.pad_size, datetime.now())
    writer = SummaryWriter('/data-output/{}'.format(TIMESTAMP))
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

    # dev_best_loss = float('inf')
    dev_best_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    sum_batch = 0
    for epoch in range(config.num_epochs):
        total_batch = 0
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        pbar = tqdm(total=len(train_iter))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            # loss = FocalLoss(gamma=config.gamma, num_class=config.num_classes)(outputs, labels)
            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # if i % 2 == 1:
            #     optimizer.step()
            #     model.zero_grad()

            optimizer.step()
            # model.zero_grad()
            
            if total_batch % 50 == 0 and total_batch != 0:

                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    # torch.save(model.state_dict(), config.save_path)
                    # torch.save(model.state_dict(), 'out/model/epoch{}_{}_pytorch_model.bin'.format(epoch, config.model_name))
                    torch.save(model.state_dict(), 'out/model/best_{}_pytorch_model.bin'.format(config.model_name))
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                # time_dif = get_time_dif(start_time)
                print('\n Iter: {},  Train Loss: {:.3f}  Train Acc: {:.3f}%  Val Loss: {:.3f}  Val Acc: {:.3f}% {} '.format(total_batch, loss.item(), train_acc * 100, dev_loss, dev_acc * 100, improve))

                writer.add_scalar('Loss/train', loss.item(), sum_batch)
                writer.add_scalar('Loss/dev', dev_loss, sum_batch)
                writer.add_scalar('Acc/train', train_acc, sum_batch)
                writer.add_scalar('Acc/dev', dev_acc, sum_batch)
                writer.flush()

                model.train()

            pbar.update(1)
            sum_batch += 1
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        pbar.close()

        true = labels.data.cpu()
        predic = torch.max(outputs.data, 1)[1].cpu()
        train_acc = metrics.accuracy_score(true, predic)
        dev_acc, dev_loss = evaluate(config, model, dev_iter)

        writer.add_scalar('Loss/train', loss.item(), sum_batch)
        writer.add_scalar('Loss/dev', dev_loss, sum_batch)
        writer.add_scalar('Acc/train', train_acc, sum_batch)
        writer.add_scalar('Acc/dev', dev_acc, sum_batch)
        writer.flush()

        print('\n Epoch{},  Train Loss: {:.3f} Train Acc: {:.3f} Val Loss: {:.3f}  Val Acc: {:.3f}  '.format(epoch + 1, loss.item(), train_acc, dev_loss, dev_acc))
        torch.save(model.state_dict(), 'out/model/{}_pytorch_model.bin'.format(config.model_name))
        
        if flag:
            break

    # torch.save(model.state_dict(), 'out/model/{}_pytorch_model.bin'.format(config.model_name))
    predict(config, model, test_iter, dev_acc)
    
    writer.close()
    # torch.save(model.state_dict(), 'pytorch_model.bin')
    # test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    # test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    test_acc, test_loss, test_report, test_confusion = evaluate(
        config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


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

    logits = pd.DataFrame(logits_all[1:])
    save_name_logits = 'out/logits_{}_{:.3f}.csv'.format(config.model_name, dev_acc)
    logits.to_csv(save_name_logits)
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
    logits = pd.DataFrame(logits_all[1:])
    save_name_logits = 'out/test/test_logits_{}.csv'.format(config.model_name)
    logits.to_csv(save_name_logits)
    if test:
        report = metrics.classification_report(
            labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)