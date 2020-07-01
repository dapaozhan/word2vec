## 代码功能
#### 1、基于gensim的word2vec模型的训练
#### 2、基于skip_gram模型的词向量模型的训练
#### 3、词向量模型的评估（近义词、反义词）
#### 4、兼容基于gensim的word2vec模型的训练、基于skip_gram模型的词向量模型的训练的增量模型的训练
#### 5、转换词向量

## 代码结构
```shell
chinese-word2vec-pytorch                    
|--Chinese_Dictionary    #存放评估语料的文件夹                  
|  |--dict_antonym.txt  #反义词                                        
|  |--pku_sim_test.txt  #近义词                                                                           
|  |--CleaningSet.json  #用于特殊字符清洗的json                                                                               
|--pyw2v                                    
|  |--callback         #skip神经网络的底层代码                         
|  |  |--lrscheduler.py                     
|  |  |--modelcheckpoint.py                 
|  |  |--progressbar.py                     
|  |  |--trainingmonitor.py                        
|  |--common          #文件处理的处理代码                      
|  |  |--tools.py                                 
|  |--config                                
|  |  |--basic_config.py  #路径配置文件                         
|  |--dataset                               
|  |  |--processed                          
|  |  |  |--vocab.pkl     #模型过程文件存放（基本没用，但是检查错误时会用）                                    
|  |  |--raw                                
|  |  |  |--zhihu.txt     #训练语料                                                  
|  |  |--stopwords.txt    #停用词                                          
|  |--ensemble                                                    
|  |--feature                                                   
|  |--io                                    
|  |  |--dataset.py           #负采样模型训练时的数据处理代码                                    
|  |--model                                 
|  |  |--nn                                 
|  |  |  |--gensim_word2vec.py  #基于gensim的word2vec模型            
|  |  |  |--skip_gram.py        #skip_gram模型                                        
|  |  |--word2vector_model200618_word2vec.pkl  #输出的词向量模型（基于gensim的word2vec模型是存放在这里）
|  |  |--word2vector_model200618_word2vec.txt
|  |  |--word2vector_model200619_word2vec.pkl
|  |  |--word2vector_model200619_word2vec.txt
|  |  |--word2vector_model_word2vec.txt            
|  |--output                                
|  |  |--checkpoints                        
|  |  |  |--word2vec.model                  
|  |  |  |--word2vec.pth                                       
|  |  |--embedding                          
|  |  |  |--dict_country.pkl                
|  |  |  |--gensim_word2vec.bin             
|  |  |  |--pytorch_word2vec2.bin   #输出的词向量模型（基于skip_gram的word2vec模型是存放在这里）                                               
|  |  |--figure                             
|  |  |  |--skip_gram_training_monitor.json                                                       
|  |--preprocessing                         
|  |  |--preprocessor.py                           
|  |--train                                 
|  |  |--trainer.py                                   
|--README.md                                
|--TrainWord2VecModel.py                    
|--TrainWord2vecSkipGram.py                     
|--Common.py     #语料预处理的python文件                           
|--Dictionary    #语料预处理用到的字符文件   
|--example.py         #代码使用样例                            
|--IncrementalModel.py   #兼容skip-gram以及基于gensim的word2vec的增量模型                   
|--ModelEstimate.py       #词向量模型评估  
|--DataTransformer.py  #转词向量代码 
```
## 执行方案
### 1、基于gensim的word2vec模型的训练
#### 词向量的长度理论上越长的话可以存储的信息越多，所以可以如果训练效果不满意，可以通过调整词向量的长度以及词窗口还有词频，调整模型，从而提高词向量模型的精度
```python
from pyw2v.config.basic_config import configs as config
from TrainWord2VecModel import TrainWord2VecModel
from Common import get_corpus

EmbeddingLen = 16# 词向量长度
MinCount = 3# 模型训练最小词数量，设置的越小，那么模型就越大，同=同时噪声也大，所以建议是在3-5之间
Window = 3# 词窗口
file1=config['data_path']#语料存放的位置
corpus= get_corpus(file1,encoding='utf-8')#语料预处理
train_model=TrainWord2VecModel(corpus=corpus,EmbeddingLen=EmbeddingLen,MinCount=MinCount,Window=Window)#执行模型
output（以下是打印的日志）:

Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\560350\AppData\Local\Temp\jieba.cache
Loading model cost 0.686 seconds.
Prefix dict has been built successfully.
1773460
E:\chinese-word2vec-pytorch\pyw2v\dataset\raw\zhihu.txt
语料【E:\chinese-word2vec-pytorch\pyw2v\dataset\raw\zhihu.txt】 词数：1773460
模型存放在E:\chinese-word2vec-pytorch\pyw2v\model\word2vector_model200619_word2vec.pkl
输出的是gensim-word2vec的模型
模型反义词平均相似度0.8410969090909092
[ 0.35355997  0.93579906 -1.0375402   0.44919413  0.4421998   1.6134751
  0.65111893 -0.94267374  0.09651817  2.6819623  -0.6016889   0.5060316
  0.69924307 -0.02305622  3.9818265  -0.23619223]
总共耗时：0.06482744216918945s
```

### 2、基于skip_gram模型的词向量模型的训练
```python
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

output（以下是打印的日志）:

seed is 2018
starting load train data from disk
模型存放在E:\chinese-word2vec-pytorch\pyw2v\output\embedding\pytorch_word2vec2.bin
read data and processing
build vocab
vocab size: 3027
initializing model
initializing callbacks
training model....
Epoch 1/6
[Training] 765/406224 [..............................] - ETA: 42:11  loss: 8.3178
。。。。。
load emebedding weights
Total 8888 word vectors.
[ 4.65063825e-02  2.23203357e-02  8.31437390e-03 -1.29481301e-01
。。。。。。
 -1.70853794e-01 -2.03937098e-01  6.63584173e-02  7.24722371e-02]
总共耗时：0.011003255844116211s
```

### 3、词向量模型的评估（近义词、反义词）
```python
from ModelEstimate import ModelEstimate
estimate=ModelEstimate(embedding_path=config['pytorch_embedding_path'],model_type='skip-gram')
#embedding_path模型的路径
estimate.similar_model_estimate()
estimate.antonym_model_estimate()

output（以下是打印的日志）:
 load emebedding weights
Total 8888 word vectors.
模型反义词平均相似度0.6725872108433731
模型反义词平均相似度0.6849341179078016
```

### 4、兼容基于gensim的word2vec模型的训练、基于skip_gram模型的词向量模型的训练的增量模型的训练
```python
from IncrementalModel import IncrementalModel
file='E:/chinese-word2vec-pytorch/pyw2v/dataset/raw/zhihu.txt'
model=IncrementalModel(before_model_type='skip-gram',before_model_path='',mincount=20,window=3,new_corpus_path=file,embeddinglen=300)
model.add_model()

#如果之前的模型是skip-gram则无需设置之前模型的路径
```python
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

output（以下是打印的日志）:
training on a 2931740 raw words (2103538 effective words) took 2.7s, 791644 effective words/s
storing 8937x300 projection weights into E:\chinese-word2vec-pytorch\pyw2v\model\word2vector_model200619_word2vec.txt
>>>>>>>>>>>>>>>>>>>>>
模型保存在E:\chinese-word2vec-pytorch\pyw2v\model\word2vector_model200619_word2vec.pkl
>>>>>>>>>>>>>>>>>>>>>
```


### 5、转换词向量
```python
data_transformer=DataTransformer(embedding_path='E:\chinese-word2vec-pytorch\pyw2v\model\word2vector_model200619_word2vec.pkl',model_type='gensim-word2vec')
#embedding_path填的是你模型的位置，model_type填的是你的模型类型，不痛的类型转换词向量的代码不同，增强模型之后的模型填gensim-word2vec
import time
time1=time.time()
print(data_transformer.get_vector('男人'))
time2 = time.time()
print('总共耗时：' + str(time2 - time1) + 's')


output（以下是打印的日志）:
[ 0.22446975 -0.193853    0.03237638 -0.23277654  0.8358967   0.06966747
  0.10515225 -1.0692025  -0.04974835  0.77911466  0.05219087  1.2675174
。。。。。。
 -0.09543055  0.2553857  -0.35091025 -0.34998843  0.35112733  0.14458749]
总共耗时：0.055848121643066406s

```


## 包版本：
jieba                              0.42.1 
scikit-learn                       0.22.1             
mpmath                             1.1.0              
torch                              1.5.0              
torchvision                        0.6.0 



微信公众号：屁屁和铭仔的数据之路
