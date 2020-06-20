"""
@Software:spyder
@File:DataTransformer.py
@Created on Thu Jun  4 16:08:47 2020
@author: betty
@description:文本转词向量
"""
#encoding:utf-8
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from math import*
from decimal import Decimal
from pyw2v.config.basic_config import configs as config
import pickle
class DataTransformer(object):
    def __init__(self,embedding_path,model_type):
        self.embedding_path = embedding_path
        self.model_type=model_type
        if self.model_type=='skip-gram':
            self.reset()
    def reset(self):
        self.load_embedding()

    # 加载词向量矩阵
    def load_embedding(self):
        print(" load emebedding weights")
        self.embeddings_index = {}
        self.words = []
        self.vectors = []
        f = open(self.embedding_path, 'r',encoding = 'utf8')
        for line in f:
            values = line.split(' ')
            try:
                word  = values[0]
                self.words.append(word)
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
                self.vectors.append(coefs)
            except:
                print("Error on ", values[:2])
        f.close()
        self.vectors = np.vstack(self.vectors)
        print('Total %s word vectors.' % len(self.embeddings_index))
        # return self.vectors
  
    # 计算相似度
    def get_words_vectors(self, word):
        if word not in self.embeddings_index:
            raise ValueError('%d not in vocab')
        current_vector = self.embeddings_index[word]
        # print('current_vector')
        # print(len(current_vector))
        # result = cosine_similarity(current_vector.reshape(1, -1), self.vectors)
        # result = np.array(result).reshape(len(self.words), )
        # idxs = np.argsort(result)[::-1][:w_num]
        # print("<<<" * 7)
        # print(word)
        # for i in idxs:
        #     print("%s : %.3f\n" % (self.words[i], result[i]))
        # print(">>>" * 7)
        return current_vector
    def get_vector(self,word):
        if self.model_type=='skip-gram':
            return self.get_words_vectors(word)
        elif self.model_type=='gensim-word2vec':
            with open(self.embedding_path,'rb') as f:
                model = pickle.load(f)            
            return model.wv.word_vec(word)
        else:
            print("人工警告：model_type输入错误，只有skip-gram和gensim-word2vec两种，别乱填")

if __name__ =="__main__":
    data_transformer=DataTransformer(embedding_path=config['pytorch_embedding_path'],model_type='skip-gram')
    import time
    time1=time.time()
    print(data_transformer.get_vector('男人'))
    time2 = time.time()
    print('总共耗时：' + str(time2 - time1) + 's')
