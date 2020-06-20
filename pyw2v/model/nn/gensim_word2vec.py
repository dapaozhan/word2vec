#encoding:utf-8
from gensim.models import word2vec
class Word2Vec():
    def __init__(self,size,
                 sg,
                 iter,
                 seed,
                 save_path,
                 num_workers,
                 window,
                 min_count):

        self.size=size
        self.sg = sg
        self.seed = seed
        self.iter = iter
        self.window = window
        self.min_count = min_count
        self.workers = num_workers
        self.save_path = save_path

    def train_w2v(self, data):
        model = word2vec.Word2Vec(data,
                                  size=self.size,
                                  window=self.window,
                                  sg=self.sg,
                                  min_count=self.min_count,
                                  workers=self.workers,
                                  seed=self.seed,
                                  compute_loss=True,
                                  iter= self.iter)
        print(model.get_latest_training_loss())
        with open(self.save_path,'w') as fw:
            for word in model.wv.vocab:
                vector = model[word]
                fw.write(str(word) + ' ' + ' '.join(map(str, vector)) + '\n')
