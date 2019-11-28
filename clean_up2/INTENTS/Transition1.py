from random import randint
from INTENTS.INTENT_FORMAT import INTENT_FORMAT
import re
from INTENTS.generalFunctions import CHINESE_STRING_TO_INT
#transition 指令需要大重构，删东西，我们怎么知道删哪个。不如在unity那边储存最后建立的物件是哪个把。
#可以在unity那边处理。

#再优化吧


class Transition1(INTENT_FORMAT):
    re_collection=[
        '个(?P<location>.*)的',
        '在(?P<location>.*)',
        '坐(?P<sit>.*)',
        '桌子(?P<desk>.*)',
        '椅子(?P<chair>.*)',
        '到(?P<location>.*)边',
        '到(?P<location>.*)',

    ]

    # check_quantity=[]

    item_converter={

        "桌子":'desk',
        "椅子":'chair',
        "椅":'chair'
    }

    location_converter={
        "正中央":(0,0),
        "左上角":(3,2),
        "右上角":(3,2),
        "左下角":(-3,-2),
        "右下角":(3,-2),
        "上":(0,2),
        "下":(0,-2),
        "左":(-2,0),
        "右":(2,0),
        "前":(0,-1),
        "后":(0,1),
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


        matched=False

        for expressions in self.re_collection:
            match_result=re.search(expressions,chinese_str)
            if not match_result is None:
                for k in match_result.groupdict().keys():
                    res_dict = match_result.groupdict()
                    val = res_dict[k]
                    data.update({k: val})
                    matched=True
                # break
        item_command=data['item_name']
        item_command=self.item_converter[item_command]
        data.update({'item_name':item_command})
        print(data)

        if 'location' in data.keys():
            try:
                val=data['location']
                x,y =self.location_converter[val]
                data.update({'x': x, 'y': y})

            except NameError:
                pass
        if 'sit' in data.keys():
            data.update({'item_name':'chair'})
        if 'desk' in data.keys():
            data.update({'item_name': 'desk'})
        if 'chair' in data.keys():
            data.update({'item_name': 'chair'})

        if matched:
            iter_time=data['quantity']
            iter_time=CHINESE_STRING_TO_INT(iter_time)
            for i in range(0,iter_time):
                res_string='Transition '+data['item_name']+' '+str(data['x'])+' '+str(data['y'])
                print(res_string)
        else:
            print('exception')