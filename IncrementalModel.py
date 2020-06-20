# -*- coding: utf-8 -*-
"""
@Software:spyder
@File:IncrementalModel.py
@Created on Thu Jun  4 16:08:47 2020
@author: betty
@description:增量模型
"""
import numpy as np
import pandas as pd
from pyw2v.config.basic_config import configs as config
from gensim.models import Word2Vec
from Common import get_corpus
import time
import pickle
class IncrementalModel():
    def __init__(self, before_model_type,before_model_path,mincount,window,new_corpus_path,embeddinglen):
        self.before_model_type=before_model_type #原先模型的类型 skip-gram或gensim-word2vec
        self.before_model_path=before_model_path #原先模型的位置，skip-gram固定存放在config['pytorch_embedding_path'] gensim-word2vec存放在pyw2v中model中txt结尾的文件
        self.mincount=mincount #词频
        self.window=window #词窗口
        self.new_corpus_path=new_corpus_path #新预料位置，原始就可以，不用清洗
        self.embeddinglen=embeddinglen #词长度，要跟你原先的向量长度一致
        
    def output_word_vector_model(self):
        embeddings_index = {}
        words = []
        vectors = []
        f = open(config['pytorch_embedding_path'], 'r',encoding = 'utf8')
        for line in f:
            values = line.split(' ')
            try:
                word  = values[0]
                words.append(word)
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
                vectors.append(coefs)
            except  Exception as e:
                print(e)
                print("Error on ", values[:2])
        df = pd.DataFrame(embeddings_index).T.reset_index(drop=False)
        L=['']*301
        L[0]=int(len(embeddings_index))
        L[1]=str(len(coefs))
        df_L= df.loc[[0],:]
        df_L.iloc[0]=L
        df=df_L.append(df)
        df.to_csv(str(config['word_vector_model'])+'_word2vec.txt',sep=' ',index=False,header=None)
    
    
    def add_model_train(self):
        '''
        Parameters
        ----------
        file : 路径
            存放新预料的路径，无需清洗。
        mincount : 词频，
            出现次数超过mincount的词语才生成向量.
        embeddinglen : 生成的向量维度
            这里的维度需要跟你原先的那个的维度一致
        return:
            新的模型
        '''
        corpus=get_corpus(self.new_corpus_path,'utf-8')
        word = list()
        for data_i in open(self.before_model_path, 'r', encoding='UTF-8'):
            data_i = data_i.split()
            if len(data_i) == self.embeddinglen + 1:
                word.append([data_i[0]]*self.mincount)
        
        # 创建Word2Vec模型训练
        model = Word2Vec(size=self.embeddinglen, window=self.window, min_count=self.mincount, iter=5)
        model.build_vocab(word)  # 先将词向量表中的词加入到模型中，才能导入其词向量
        model.intersect_word2vec_format(str(config['word_vector_model'])+'_word2vec.txt', lockf=1.0, binary=False)#加入已有词向量
        # ####################################### 增量训练
        model.build_vocab(corpus, keep_raw_vocab=True, update=True)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
        out_model_path=str(config['word_vector_model'])+str(time.strftime('%y%m%d'))
        model.wv.save_word2vec_format(out_model_path+'_word2vec.txt', binary=False)  
        with open(out_model_path+'_word2vec.pkl', 'wb') as f:
            pickle.dump(model, f)
        print('>>>'*16)
        print('再训练的模型保存在'+out_model_path+'_word2vec.pkl')
        print('>>>'*16)
    def add_model(self):
        if self.before_model_type=='skip-gram':
            self.output_word_vector_model()
            self.before_model_path=str(config['word_vector_model'])+'_word2vec.txt'
            self.add_model_train()
        elif self.before_model_type=='gensim-word2vec':
            self.add_model_train()
        else:
            print('输入的before_model_type有误，请重新输入')
        
if __name__ =="__main__":
    file='E:/chinese-word2vec-pytorch/pyw2v/dataset/raw/zhihu.txt'
    model=IncrementalModel(before_model_type='skip-gram',before_model_path='',mincount=20,window=3,new_corpus_path=file,embeddinglen=300)
    
    before_model_path='E:\chinese-word2vec-pytorch\pyw2v\model\word2vector_model200619_word2vec.txt'
    model=IncrementalModel(before_model_type='gensim-word2vec',before_model_path=before_model_path,mincount=20,window=3,new_corpus_path=file,embeddinglen=300)
    model.add_model()

    



    
    
