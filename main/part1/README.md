# Bert-Chinese-Text-Classification-Pytorch
中文文本分类，Bert，ERNIE

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn 
tensorboardX


## 效果

### Bert base

- (3, 128, 32, 5e-5, O1, (0.835, 0.778))
- (4, 128, 32, 5e-5, O1, (0.858, **0.784**))
- (5, 128, 32, 5e-5, O1, (0.858, 0.779))

### Bert base + CNN

- (3, 128, 32, 5e-5, O1, (0.858, **0.783**))
- (4, 128, 32, 5e-5, O1, (0.858, 0.781))

### ERNIE

- pt (3, 32, 128, 8e-5, O1, (0.968, 0.791)
- focal (3, 32, 128, 8e-5, O1, 2, (0.968, 0.793)) (0.796 max) 58_34
- focal (3, 32, 128, 8e-5, O1, 2.5, (0.935, 0.783))
- focal (3, 64, 128, 8e-5, O1, 2, (0.935, 0.792)) 13_19
- focal (4, 64, 128, 7e-5, O1, 2, (0.857, **0.796**)) 23_02
  - shuffle focal pt (4, 64, 128, 7e-5, O1, 2, (0.805, 0.784)) 41_32
- focal (5, 64, 128, 7e-5, O1, 2, (0.889, 0.782)) 30_27
- focal (4, 64, 128, 9e-5, O1, 2, (0.840, 0.790)) 42_15
- focal (4, 64, 128, 6e-5, O1, 2, (0.889, 0.789)) 49_15
- focal (5, 64, 128, 7e-5, O1, 2, (0.889, 0.782)) 16_15
- focal (5, 64, 128, 6e-5, O1, 2, (0.873, 0.789)) 24_47
- focal (5, 64, 128, 5e-5, O1, 2, (0.889, 0.792)) 32_52
- focal (5, 64, 128, 4e-5, O1, 2, (0.873, 0.786)) 41_57
- focal pt (3, 32, 128, 8e-5, O1, 1.75, (0.935, 0.790))
- focal pt (3, 32, 128, 8e-5, O1, 2.25, (0.935, 0.791))
  - no shuffle focal pt (3, 32, 128, 8e-5, O1, 2, (1, 0.791)) (0.795 max)
  - shuffle focal pt (3, 32, 128, 8e-5, O1, 2, (0.733, 0.791))
- pt (3, 32, 128, 2e-5, O1, (0.968, 0.787))
- (4, 128, 32, 5e-5, O1, (0.858, 0.787)) 
- pt (4, 128, 32, 5e-5, O1, (0.866, 0.786))
- pt (4, 128, 32, 8e-5, O1, (0.866, 0.787))
- pt0 (4, 128, 32, 8e-5, O1, (0.835, 0.782)) epoch0 not good
- (5, 128, 32, 5e-5, O1, (0.843, 0.774)) 
- pt (5, 128, 32, 2e-5, O1, (0.858, 0.778))
- pt (5, 128, 32, 5e-5, O1, (0.890, 0.783))
- pt (5, 128, 32, 8e-5, O1, (0.850, 0.789))
- pt (5, 128, 32, 1e-4, O1, (0.827, 0.789))
- (ag, pt) (5, 128, 32, 8e-5, O1, (0.955, 0.783))
- (ag, pt) (3, 128, 32, 8e-5, O1, (0.864, 0.780))

### Roberta base

- (4, 128, 32, 5e-5, O1, (0.850, 0.775))
- (5, 128, 32 ,2e-5, O1, (0.858, 0.785))
- (5, 128, 64 ,2e-5, O1, (0.866, **0.785**))
- (5, 64, 128 ,2e-5, O1, (0.873, 0.782))
- (3, 64, 128 ,2e-5, O1, (0.889, 0.781))

### Roberta large

- (3, 32, 32, 2e-5, O1, (0.935, 0.784))
- (3, 16, 128, 2e-5, O1, (0.933, **0.789**))

## 预训练语言模型
bert模型放在 bert_pretain目录下，ERNIE模型放在ERNIE_pretrain目录下，每个目录下都是三个文件：
 - pytorch_model.bin  
 - bert_config.json  
 - vocab.txt  

预训练模型下载地址：  
bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz  
              词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt  
来自[这里](https://github.com/huggingface/pytorch-transformers)   
备用：模型的网盘地址：https://pan.baidu.com/s/1qSAD5gwClq7xlgzl_4W3Pw

ERNIE_Chinese: http://image.nghuyong.top/ERNIE.zip  
来自[这里](https://github.com/nghuyong/ERNIE-Pytorch)  
备用：网盘地址：https://pan.baidu.com/s/1lEPdDN1-YQJmKEd_g9rLgw  

解压后，按照上面说的放在对应目录下，文件名称确认无误即可。  

## 使用说明
下载好预训练模型就可以跑了。
```
# 训练并测试：
# bert
python run.py --model bert

# bert + 其它
python run.py --model bert_CNN

# ERNIE
python run.py --model ERNIE
```
