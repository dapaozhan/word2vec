# -*- coding: utf-8 -*-
"""
@Software:spyder
@File:Common.py
@Created on Thu Jun  4 16:08:47 2020
@author: betty
@description:文本清洗
"""
import jieba
import os
import json
import re


# 文本清洗 类
class TextClean(object):
    # 初始化
    def __init__(self, cleaning_set):

        # 判断是否输入清洗规则路径 .json 文件
        if cleaning_set is not None:
            # 提取用户词典文件绝对路径
            if os.path.isfile(cleaning_set):  # 输入绝对路径
                self.cleaning_set_path = cleaning_set
            else:  # 输入文件名，则默认其存放路径 为 .\NLP\Dictionary
                #  当前文件路径 的上层路径， 'NLP' 所在目录   'C:\Users\Vincent\Desktop\NLP'
                cwd = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.'))
                self.cleaning_set_path = os.path.join(cwd, 'Dictionary', cleaning_set)

            # 读取文本清洗设置信息
            with open(self.cleaning_set_path, 'r', encoding='utf8') as load_f:
                cleaning_set = json.load(load_f)

            #  【删除】 正则表达式列表
            self.re_clear = list()
            for re_del in cleaning_set['delete']:
                self.re_clear.append(re.compile(re_del))

            #  【替换】 正则表达式列表
            self.re_sub = list()
            for re_sub_new, re_sub_old, comment in cleaning_set['replace']:
                self.re_sub.append([re_sub_new, re.compile(re_sub_old), comment])

        else:
            self.cleaning_set_path = None

    # 清洗文本
    def __call__(self, text):
        # 转换成小写
        text = text.lower()

        if self.cleaning_set_path is not None:
            # 文本清洗 删除无效字符
            for re_clear_i in self.re_clear:
                text = re.sub(re_clear_i, '', text)

            # 文本替换 用指定字符替换
            for re_sub_new, re_sub_old, _ in self.re_sub:
                text = re.sub(re_sub_old, re_sub_new, text)

        return text

    # 打印正则清洗规则
    def __repr__(self):
        return '【re_clear】:{}\n【re_sub】:{}\n'.format(self.re_clear, self.re_sub)


# 文本切词 类
class TextTokenizer(object):

    # 初始化
    def __init__(self, user_word=None, stop_word=None):
        #  当前文件路径 的上层路径， 'NLP' 所在目录   'C:\Users\Vincent\Desktop\NLP'
        cwd = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.'))

        # 判断是否输入用户词典
        if user_word is not None:
            # 提取用户词典文件绝对路径
            if os.path.isfile(user_word):  # 输入绝对路径
                self.user_word_path = user_word
            else:  # 输入文件名，则默认其存放路径 为 .\NLP\Dictionary
                self.user_word_path = os.path.join(cwd, 'Dictionary', user_word)

            # 添加用户词典
            jieba.load_userdict(self.user_word_path)

        else:
            self.user_word_path = None

        # 判断是否输入停用词典
        if stop_word is not None:
            # 提取停用词典文件绝对路径
            if os.path.isfile(stop_word):  # 输入绝对路径
                self.stop_word_path = stop_word
            else:  # 输入文件名，则默认其存放路径 为 .\NLP\Dictionary
                self.stop_word_path = os.path.join(cwd, 'Dictionary', stop_word)

            # 读取停用词典
            stop_word = ['\n', ' ']
            for line_i in open(self.stop_word_path, 'rb'):
                line_i = line_i.strip()

                try:  # 若文件不是以 uft-8 格式存储将报错。
                    stop_word.append(line_i.decode('utf-8').lstrip('\ufeff'))
                except UnicodeDecodeError:
                    raise ValueError('StopWord File 【%s】 must be utf-8 !' % self.stop_word_path)

            self.stop_word = stop_word

        else:
            self.stop_word_path = None
            self.stop_word = []

    # 分词模型
    def __call__(self, corpus):
        # jieba 分词
        seg_list = jieba.lcut(corpus, cut_all=False)

        # 去掉停用词
        select_seg_list = [seg_i for seg_i in seg_list if seg_i not in self.stop_word]

        return select_seg_list

    # 打印分词类信息
    def __repr__(self):
        return 'UserWordPath:{}\nStopWordPath:{}\nStopWord:{}\n'.\
            format(self.user_word_path, self.stop_word_path, self.stop_word)


# 获取文本数据 并清洗 切词
def get_corpus(file,encoding):
    # 初始化清洗类 模块
    cleaning = TextClean(cleaning_set="CleaningSet.json")
    # 初始化切词类 模块
    cutting = TextTokenizer(user_word="UserWord.txt", stop_word="StopWord.txt")

    corpus_token = list()
    corpus_len = 0
    for i, corpus_i in enumerate(open(os.path.join('.', 'Data', file), 'r', encoding=encoding)):

        corpus_len += len(corpus_i)

        # 清洗 切词
        token = cutting(cleaning(corpus_i))

        if len(token) > 0:
            # 切词后再以空格组合在一起
            corpus_token.append(token)
    print(corpus_len)
    print(file)
    print("语料【{:12s}】 词数：{}".format(str(file), str(corpus_len)))
    return corpus_token
