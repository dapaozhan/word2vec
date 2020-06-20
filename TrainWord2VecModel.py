# -*- coding: utf-8 -*-
"""
@Software:spyder
@File:TrainWord2VecModel.py
@Created on Thu Jun  4 16:08:47 2020
@author: betty
@description:基于gensim的word2vec模型训练
"""
from ModelEstimate import ModelEstimate
from Common import get_corpus
from pyw2v.config.basic_config import configs as config
import pickle
from gensim.models import Word2Vec
import time
class TrainWord2VecModel():
    def __init__(self,corpus,EmbeddingLen,MinCount,Window):
        self.corpus=corpus
        self.EmbeddingLen=EmbeddingLen
        self.MinCount=MinCount
        self.Window=Window
        self.run_model()
        
    def run_model(self):
        self.train()
        self.output_model()
        self.estimate_model()
        
    def train(self):
        # 创建Word2Vec模型训练
        self.model = Word2Vec(size=self.EmbeddingLen, window=self.Window, min_count=self.MinCount, iter=5)
    
        self.model.build_vocab(self.corpus)
        self.model.train(self.corpus, total_examples=self.model.corpus_count, epochs=self.model.iter)
        # print("语向量维度：{}".format(self.model.wv.syn0.shape))
        # print("【开心】相似词：")
        # print("  ".join(["{}: {:.2f}".format(word_i, s_i) for word_i, s_i in self.model.wv.similar_by_word("开心", topn=20)]))
    
        # # 模型中词的信息
        # vocab_info = [[word_i, value_i.count, value_i.index]for word_i, value_i in model.wv.vocab.items()]
        # # 模型中前 20 个词
        # print(sorted(vocab_info, key=lambda x: x[2], reverse=False)[:20])
    def output_model(self):
        out_model_path=str(config['word_vector_model'])+str(time.strftime('%y%m%d'))
        print('模型存放在'+out_model_path+'_word2vec.pkl')
        print('输出的是gensim-word2vec的模型')
        with open(out_model_path+'_word2vec.pkl', 'wb') as f:
            pickle.dump(self.model, f)
    def estimate_model(self):
        out_model_path=str(config['word_vector_model'])+str(time.strftime('%y%m%d'))
        model_path=out_model_path+'_word2vec.pkl'
        estimate=ModelEstimate(embedding_path=model_path,model_type='gensim-word2vec')
        # 以下是模型效果
        estimate.similar_model_estimate()
        estimate.antonym_model_estimate()
if __name__ =="__main__":
    # 词向量长度
    EmbeddingLen = 16
    # 模型训练最小词数量
    MinCount = 3
    # 词窗口
    Window = 3
    file1=config['data_path']
    corpus= get_corpus(file1,encoding='utf-8')
    train_model=TrainWord2VecModel(corpus=corpus,EmbeddingLen=EmbeddingLen,MinCount=MinCount,Window=Window)
    



    # model_path=out_model_path+'_word2vec.pkl'
    # estimate=ModelEstimate(embedding_path=model_path,model_type='gensim-word2vec')
    # estimate.similar_model_estimate()
    # estimate.antonym_model_estimate()
    
    