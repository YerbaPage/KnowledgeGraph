import os

# 启动 neo4j
# os.system('/root/neo4j-community-4.2.2/bin/neo4j start')

import requests
import flask
from flask import Flask

app = Flask(__name__)

model = load_weights(path)

@app.route('/')
def main():
    pass

@app.route('/model1')
def model1(data):
    # 加载训练好的模型参数
    # load_weights
    results = model.predict(data)
    # 输入数据、返回结果
    return results


@app.route('/')
def main(information):
    
    data = a(information)
    results = requests.post('localhost://port/model1', data=data)
    
    return res





