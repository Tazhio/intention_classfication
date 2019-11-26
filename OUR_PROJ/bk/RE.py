import argparse
from enum import Enum,unique
import re
import numpy as np
from random import randint
import random
import os
import time
@unique
class INTENT_TYPE(Enum):
    INSTANTIATION=1
    DELETE=2
    TRANSICTION=3
    CHANGE_COLOR=4


TYPE_NAME={
    INTENT_TYPE.INSTANTIATION:"放置物品",
    INTENT_TYPE.DELETE:"删除物品",
    INTENT_TYPE.TRANSICTION:"移动物品",
    INTENT_TYPE.CHANGE_COLOR:"改变颜色"

}

class INTENT_FORMAT:
    def __init__(self):
        # self.type=type_
        # self.re_collection=re_collection
        return
    def write_file(self):
        raise NotImplementedError

class Instantiation(INTENT_FORMAT):
    re_collection=[
        '.*需要(?P<quantity>[零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬1234567890]+)个(?P<item_name>.*)',
        '(？P<item_name>.*)在哪里',
    ]

    item_converter={
        "桌子":'desk',
        "椅子":'chair',
        "椅":'chair'
    }

    def __init__(self ):
        super().__init__()

    def analyze(self,chinese_str):
        data={
            'quantity': 1,
            'x': 0,
            'y': 0,
            'item_name': "桌子"
        }
        x=randint(-7,7)
        y=randint(-5,5)
        data.update({'x':x,'y':y})

        matched=False

        for expressions in self.re_collection:
            match_result=re.search(expressions,chinese_str)
            if not match_result is None:
                for k in match_result.groupdict().keys():
                    res_dict = match_result.groupdict()
                    val = res_dict[k]
                    data.update({k: val})
                    matched=True
                break
        item_command=data['item_name']
        item_command=self.item_converter[item_command]
        data.update({'item_name':item_command})

        if matched:
            iter_time=data['quantity']
            iter_time=Chinese_String_to_int(iter_time)
            for i in range(0,iter_time):
                res_string='Instantiation '+ str(data['x'])+' '+str(data['y'])+' '+item_command
                print(res_string)
        else:
            print('exception')



def info_extraction_re(text:str,intent:int):


    return

chinese_int={
    '零':0,
    '一':1,
    '二':2,
    '三':3,
    '四':4,
    '五':5,
    '六':6,
    '七':7,
    '八':8,
    '九':9,
    '十':10,
    '百':100,
    '千':1000,
    '万':10000,
    '亿':1000000000,
    '壹':1,
    '贰':2,
    '叁':3,
    '肆':4,
    '伍':5,
    '陆':6,
    '柒':7,
    '捌':8,
    '玖':9,
    '拾':10,
    '佰':100,
    '仟':1000,
    '萬':10000


}
class Node():

    def __init__(self,val=None,next=None):
        self.val=val
        self.next=next
        return

class Heap():
    def __init__(self):
        self.headSentinel=Node()
        self.tailSentinel=Node()
        self.headSentinel.next=self.tailSentinel
        self.size=0
        return


    def push(self, val:int):
        n=Node(val)
        n.next = self.headSentinel.next
        self.headSentinel.next=n
        self.size=self.size+1
        return
    def pop(self):
        n=self.headSentinel.next
        self.headSentinel.next=self.headSentinel.next.next
        self.size=self.size-1
        return n.val

    def isEmpty(self):
        return self.size==0

    def seeTop(self)->int :
        va=self.headSentinel.next
        value=va.val
        return value



def Chinese_String_to_int(chinese_str:str):
    if type(chinese_str) is int:
        return chinese_str
    try:
        if type(int(chinese_str)) is int:
            return int(chinese_str)
    except ValueError:
        pass
    heap=Heap()
    chinese_str=str.strip(chinese_str)
    for ele in chinese_str:
        this_int=chinese_int[ele]
        if not heap.isEmpty():
            if heap.seeTop()< this_int :
                s=heap.pop()
                heap.push(this_int*s)
            else:
                heap.push(this_int)
        else:heap.push(this_int)
    res=0

    while not heap.isEmpty():
        res=res+heap.pop()
    return res



INSTANCIATION=["给"]

if __name__=='__main__':


    # 测试文本
    test = '<h1>hello 你好, world 世界</h1>'

    # 中文匹配正则
    chinese_pattern = '[\u4e00-\u9fa5]+'
    says = re.findall(chinese_pattern, test)
    # 输出提取的内容
    hi = ''
    for say in says:
        # print(say)
        hi += say + ','
    hi = hi.strip(',')

    # 打印结果：你好,世界
    print(hi)

    str2 = '我需要十一个桌子 1'
    res = re.search('.*需要(?P<quantity>[零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬1234567890]+)个(?P<item_name>.*)', str2)
    print(res.groupdict())

    str3 = '椅子在哪里'
    res2 = re.search('(?P<item_name>.*)在哪里', str3)
    print(res2.groupdict())

    s = '1102231990xxxxxxxx'
    res = re.search('(?P<province>\d{3})(?P<city>\d{3})(?P<born_year>\d{4})', s)
    print(res.groupdict())
    ress=Chinese_String_to_int("一百一十二")
    print(ress)

    instant=Instantiation()
    instant.analyze('我需要十一个桌子')


