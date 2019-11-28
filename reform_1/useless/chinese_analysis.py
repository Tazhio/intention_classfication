# coding=utf-8
# -*- coding: cp936 -*-
import jieba
import jieba.posseg as pseg
import codecs
import re
import os
import time
import string
from nltk.probability import FreqDist
open=codecs.open


#jieba 分词可以将我们的自定义词典导入，格式 “词” “词性” “词频”
# jieba.load_userdict('data/userdict.txt')

#定义一个keyword类
class keyword(object):
    def Chinese_Stopwords(self):          #导入停用词库
        stopword=[]
        cfp=open('data/combined_stopwords.txt','r+','utf-8')   #停用词的txt文件
        for line in cfp:
            for word in line.split():
                stopword.append(word)
        cfp.close()
        return stopword