# data={
#     'a':1,
#     'b':2
# }
#
# print(data[c])

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
import jieba


seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))


def cleaning(sentences):
    words = []
    for s in sentences:
        # clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        # w = word_tokenize(clean)
        # stemming
        seg_list = jieba.cut_for_search(s)  # 搜索引擎模式
        print(", ".join(seg_list))
        # words.append([i.lower() for i in w])
        seg_list2=", ".join(seg_list)

        print("seg2:",seg_list2)
        words.append(seg_list2)
        # print('this is words')
        # print(words)

    return words



def create_tokenizer(words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters=filters)
    token.fit_on_texts(words)
    return token

cleaned_words = cleaning(sentences)

word_tokenizer = create_tokenizer(cleaned_words)
# vocab_size = len(word_tokenizer.word_index) + 1
# max_length = max_length(cleaned_words)


def predictions(text):
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    test_word = word_tokenize(clean)
    test_word = [w.lower() for w in test_word]
    test_ls = word_tokenizer.texts_to_sequences(test_word)
    print(test_word)
    # Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))


predictions("young doesn't learn the good things")

