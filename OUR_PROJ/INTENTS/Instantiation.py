from random import randint
from INTENTS.INTENT_FORMAT import INTENT_FORMAT
import re
from INTENTS.generalFunctions import CHINESE_STRING_TO_INT


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
            iter_time=CHINESE_STRING_TO_INT(iter_time)
            for i in range(0,iter_time):
                res_string='Instantiation '+ str(data['x'])+' '+str(data['y'])+' '+item_command
                print(res_string)
        else:
            print('exception')