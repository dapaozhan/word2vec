#encoding:utf-8
from pathlib import Path
BASE_DIR = Path('pyw2v')

configs = {
    'data_path': BASE_DIR / 'E:/chinese-word2vec-pytorch/pyw2v/dataset/raw/zhihu.txt',#语料的位置
    'model_save_path': BASE_DIR / 'E:/chinese-word2vec-pytorch/pyw2v/output/checkpoints/word2vec.pth',#skip-gram模型存放的位置
    'word_vector_model':BASE_DIR /'E:/chinese-word2vec-pytorch/pyw2v/model/word2vector_model',
    'vocab_path': BASE_DIR / 'E:/chinese-word2vec-pytorch/pyw2v/dataset/processed/vocab.pkl', # 
    'pytorch_embedding_path': BASE_DIR / 'E:/chinese-word2vec-pytorch/pyw2v/output/embedding/pytorch_word2vec2.bin',#skip-gram模型词向量模型的位置
    'gensim_embedding_path':BASE_DIR / 'E:/chinese-word2vec-pytorch/pyw2v/output/embedding/gensim_word2vec.bin',

    'log_dir': BASE_DIR / 'E:/chinese-word2vec-pytorch/pyw2v/output/log',
    'figure_dir': BASE_DIR / 'E:/chinese-word2vec-pytorch/pyw2v/output/figure',
    'stopword_path': BASE_DIR / 'E:/chinese-word2vec-pytorch/pyw2v/dataset/stopwords.txt',#存放stopword的路径
    'pku_sim_path': BASE_DIR /'E:/chinese-word2vec-pytorch/Chinese_Dictionary/pku_sim_test.txt',#存放近义词的位置
    'dict_antonym_path': BASE_DIR / 'E:/chinese-word2vec-pytorch/Chinese_Dictionary/dict_antonym.txt'#存放反义词的位置
}
