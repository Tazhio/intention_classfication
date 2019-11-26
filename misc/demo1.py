# -*- coding: UTF-8 -*-
from enum import Enum, unique

import jieba

#origin webpage: https://blog.csdn.net/android_ruben/article/details/78047311

if __name__=="__main__":
    #TODO: PREP DATA

    list_sen = ['今天这个菜真好吃！', '嗨！今天天气不错！', '今天很开心，明天见！']
    #dict_voc
    dict_and_freq = dict()
#TODO: EXTRACT FEATURE
    for s in list_sen:
        for w in s:
            #python 会自动地把字符串按中文字符给割开，niubi
            if w in dict_and_freq.keys():
                dict_and_freq[w] += 1
            else:
                dict_and_freq[w] = 1
    features_one = dict()  # 吃饭  {字：频率}
    features_two = dict()  # 打招呼
    features_three = dict()  # 再见
    for w in list_sen[0]:
        if w in features_one.keys():
            features_one[w] += 1
        else:
            features_one[w] = 1
    for w in list_sen[1]:
        if w in features_two.keys():
            features_two[w] += 1
        else:
            features_two[w] = 1
    for w in list_sen[2]:
        if w in features_three.keys():
            features_three[w] += 1
        else:
            features_three[w] = 1
    print(features_one)
    print(features_two)
    print(features_three)


    sen = '今天的工作就到这里吧，大家早点回去，明天再继续吧。'
    score1 = 0.0
    score2 = 0.0
    score3 = 0.0
    print('class1:')
    for w in sen:
        if w in features_one:
            print('word:', w,'feature_one:', features_one[w], dict_and_freq[w])
            score1 += features_one[w] * dict_and_freq[w]
    print('score1:', score1)
    print('---------------------')



    print('class2:')
    for w in sen:
        if w in features_two:
            print('word', w, features_two[w], dict_and_freq[w])
            score2 += features_two[w] * dict_and_freq[w]
    print('score2:', score2)
    print('---------------------')



    print('class3:')
    for w in sen:
        if w in features_three:
            print('word', w, features_three[w], dict_and_freq[w])
            score3 += features_three[w] * dict_and_freq[w]
    print('score3:', score3)
    print('---------------------')



#TODO: 数据准备
intention_type={

}


@unique
class INTENT_TYPE(Enum):

    MOVE = 0
    ROTATE = 1  # 旋转
    CHANGE_COLOR = 2
    CHANGE_MODEL = 3
    ADD_ITEM = 4
    DELETE_ITEM = 5
    FOLD=6
    TURN_ON=7
    TURN_OFF=8
    PLAY_INSTRUMENT=9
    # Sat = 6

TYPE_NAME={
INTENT_TYPE.MOVE: "移动",
INTENT_TYPE.ROTATE: "旋转",
INTENT_TYPE.CHANGE_COLOR: "切换颜色",
INTENT_TYPE.CHANGE_MODEL: "切换模型",
INTENT_TYPE.ADD_ITEM: "增加物体",
INTENT_TYPE.DELETE_ITEM: "删除物体",
INTENT_TYPE.FOLD: "折叠",
INTENT_TYPE.TURN_ON: "打开电器",
INTENT_TYPE.TURN_OFF: "关闭电器",
INTENT_TYPE.PLAY_INSTRUMENT: "乐器演奏",
}


class feature:
    def __init__(self, _type:INTENT_TYPE):
        self._type=_type
        assert type(_type) == INTENT_TYPE




        return

#TODO: 特征提取

#TODO: 模型准备

#TODO: 训练模型

#TODO: 使用模型


