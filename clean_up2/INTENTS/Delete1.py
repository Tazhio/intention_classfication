from random import randint
from INTENTS.INTENT_FORMAT import INTENT_FORMAT
import re
from INTENTS.generalFunctions import CHINESE_STRING_TO_INT



class Delete(INTENT_FORMAT):
    re_collection=[
        '个(?P<location>.*)的',
        '在(?P<location>.*)',
        '坐(?P<sit>.*)',
        '桌子(?P<desk>.*)',
        '椅子(?P<chair>.*)',
    ]

    item_converter={

        "桌子":'desk',
        "椅子":'chair',
        "椅":'chair'
    }

    location_converter={
        "正中央":(0,0),
        "左上角":(-6,4),
        "右上角":(6,4),
        "左下角":(-6,-4),
        "右下角":(6,-4)
    }

    def __init__(self):
        super().__init__()

    def analyze(self,chinese_str):
        data={
            'item_name':'桌子',
            'item_id':0,
            'quantity':1,
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

        if 'sit' in data.keys():
            data.update({'item_name':'chair'})
        if 'desk' in data.keys():
            data.update({'item_name':'desk'})
        if 'chair' in data.keys():
            data.update({'item_name':'chair'})

        if matched:
            iter_time=data['quantity']
            iter_time=CHINESE_STRING_TO_INT(iter_time)
            for i in range(0,iter_time):
                res_string='Delete '+ str(data['item_name'])+' '+str(data['item_id'])
                print(res_string)
        else:
            print('exception')
        return
