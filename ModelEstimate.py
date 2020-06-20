# -*- coding: utf-8 -*-
"""
@Software:spyder
@File:ModelEstimate.py
@Created on Thu Jun  4 16:08:47 2020
@author: betty
@description:模型评估
"""
import re
from pyw2v.config.basic_config import configs as config
from math import*
from DataTransformer import DataTransformer
class ModelEstimate():
    def __init__(self,embedding_path,model_type):
        self.result_total_antonym=[]
        self.result_total_similar=[]
        self.data_transformer=DataTransformer(embedding_path = embedding_path,model_type=model_type)
    def square_rooted(self,x):
        return round(sqrt(sum([a*a   for a in x])),6)
    def cosinesimilarity(self,x,y):
        numerator = sum(a*b for a,b in zip(x,y))
        denominator = self.square_rooted(x)*self.square_rooted(y)
        return round(numerator/float(denominator),6)
    
    def similar_model_estimate(self):
        with open(config['pku_sim_path'], 'r', encoding='utf-8') as fin:
            for line in fin:
                try:
                    str_tmp1, str_tmp2 = line.split('\t', 1)
                    str1 = str_tmp1
                    str2 = str_tmp2.replace('\n', '')
                    result=self.cosinesimilarity(self.data_transformer.get_vector(str1),self.data_transformer.get_vector(str2))
                    self.result_total_similar.append(result)
                except :
                    pass
            print('模型反义词平均相似度'+str(sum(self.result_total_similar)/len(self.result_total_similar)))
    
    
    def antonym_model_estimate(self):
        with open (config['dict_antonym_path'],encoding='utf-8') as f:
            for line in f:
                try:
                    # print(line)
                    line=line.replace('——','--')
                    line=line.replace('―','--')
                    str_tmp1, str_tmp2 =  line.split('--',1)
                    str_tmp2 = str_tmp2.replace('\n', '')
                    result=self.cosinesimilarity(self.data_transformer.get_vector(str_tmp1), self.data_transformer.get_vector(str_tmp2))
                    self.result_total_antonym.append(result)
                except :
                      pass
            print('模型反义词平均相似度'+str(sum(self.result_total_antonym)/len(self.result_total_antonym)))

if __name__ =="__main__":
    estimate=ModelEstimate(embedding_path=config['pytorch_embedding_path'],model_type='skip-gram')
    estimate.similar_model_estimate()
    estimate.antonym_model_estimate()
   
    


