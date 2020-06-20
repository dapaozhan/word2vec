# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:20:07 2020

@author: 560350
"""



# 1、基于gensim的word2vec模型的训练
import os
os.chdir(r'E:\chinese-word2vec-pytorch')
from pyw2v.config.basic_config import configs as config
from TrainWord2VecModel import TrainWord2VecModel
from Common import get_corpus
from DataTransformer import DataTransformer 
EmbeddingLen = 16# 词向量长度
MinCount = 3# 模型训练最小词数量，设置的越小，那么模型就越大，同=同时噪声也大，所以建议是在3-5之间
Window = 3# 词窗口
file1=config['data_path']#语料存放的位置
corpus= get_corpus(file1,encoding='utf-8')#语料预处理
train_model=TrainWord2VecModel(corpus=corpus,EmbeddingLen=EmbeddingLen,MinCount=MinCount,Window=Window)#执行模型
# 以下是预测词向量
data_transformer=DataTransformer(embedding_path='E:\chinese-word2vec-pytorch\pyw2v\model\word2vector_model200619_word2vec.pkl',model_type='gensim-word2vec')
#embedding_path填的是你模型的位置，model_type填的是你的模型类型，不痛的类型转换词向量的代码不同，增强模型之后的模型填gensim-word2vec
import time
time1=time.time()
print(data_transformer.get_vector('男人'))
time2 = time.time()
print('总共耗时：' + str(time2 - time1) + 's')

# 2、基于skip_gram模型的词向量模型的训练

from TrainWord2vecSkipGram import main 
from DataTransformer import DataTransformer 
main()
#模型训练，需要更改参数，可以打开TrainWord2vecSkipGram.py文件，其中每个函数的含义有相应的注释，主要调整词频min_freq、词窗口window_size、以及词长度embedd_dim

data_transformer=DataTransformer(embedding_path=config['pytorch_embedding_path'],model_type='skip-gram')

import time
time1=time.time()
print(data_transformer.get_vector('男人'))
time2 = time.time()
print('总共耗时：' + str(time2 - time1) + 's')



from IncrementalModel import IncrementalModel
file='E:/chinese-word2vec-pytorch/pyw2v/dataset/raw/zhihu.txt'
model=IncrementalModel(before_model_type='skip-gram',before_model_path='',mincount=20,window=3,new_corpus_path=file,embeddinglen=300)
model.add_model()

#如果之前的模型是skip-gram则无需设置之前模型的路径
before_model_path='E:\chinese-word2vec-pytorch\pyw2v\model\word2vector_model200619_word2vec.txt'
model=IncrementalModel(before_model_type='gensim-word2vec',before_model_path=before_model_path,mincount=20,window=3,new_corpus_path=file,embeddinglen=300)
'''
self.before_model_type=before_model_type #原先模型的类型 skip-gram或gensim-word2vec
self.before_model_path=before_model_path #原先模型的位置，skip-gram固定存放在config['pytorch_embedding_path'] gensim-word2vec存放在pyw2v中model中txt结尾的文件
self.mincount=mincount #词频
self.window=window #词窗口
self.new_corpus_path=new_corpus_path #新预料位置，原始就可以，不用清洗
self.embeddinglen=embeddinglen #词长度，要跟你原先的向量长度一致
'''
model.add_model()
#如果之前的模型是gensim-word2vec 需设置之前模型的路径





from ModelEstimate import ModelEstimate
estimate=ModelEstimate(embedding_path=config['pytorch_embedding_path'],model_type='skip-gram')
estimate.similar_model_estimate()
estimate.antonym_model_estimate()












