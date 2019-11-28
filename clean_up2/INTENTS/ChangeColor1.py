from random import randint
from INTENTS.INTENT_FORMAT import INTENT_FORMAT
import re
from INTENTS.generalFunctions import CHINESE_STRING_TO_INT


class ChangeColor1(INTENT_FORMAT):
    re_collection=[
        '坐(?P<sit>.*)',
        '(?P<color>.)色',
        '桌子(?P<desk>.*)',
        '椅子(?P<chair>.*)',


    ]

    # check_quantity=[]
    color_rgb={
        "红":(255, 0, 0),
        "黄":(238, 255, 0),
        "蓝":(0, 106, 255),
        "绿":(238, 255, 0),
        "青":(0, 255, 153),
        "橙":(255, 187, 0),
        "紫":(72, 0, 255),
        "群青":(0, 72, 255),
        "赤":(255, 0, 68),
        "草地":(157, 255, 0),
        "棕":(128, 66, 0),
        "黑":(0,0,0),
        "白":(255,255,255),
        "粉":(255, 102, 255),

    }

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

    def __init__(self ):
        super().__init__()

    def analyze(self,chinese_str):
        data={
            'quantity': 1,
            'x': 0,
            'y': 0,
            'item_name': "桌子",
            'red':255,
            'blue':245,
            'green':255,
            'alpha':80

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
                # break
        item_command=data['item_name']
        item_command=self.item_converter[item_command]
        data.update({'item_name':item_command})
        print(data)

        # if 'location' in data.keys():
        #     try:
        #         val=data['location']
        #         x,y =self.location_converter[val]
        #         data.update({'x': x, 'y': y})
        #
        #     except NameError:
        #         pass
        if 'sit' in data.keys():
            data.update({'item_name': 'chair'})
        if 'desk' in data.keys():
            data.update({'item_name': 'desk'})
        if 'chair' in data.keys():
            data.update({'item_name': 'chair'})
        if 'color' in data.keys():
            try:
                val = data['color']
                red, green, blue = self.color_rgb[val]
                data.update({'red':red,'green':green,'blue':blue})
                # data.update({'x': x, 'y': y})
            except NameError:
                pass



        if matched:
            iter_time=data['quantity']
            iter_time=CHINESE_STRING_TO_INT(iter_time)
            # item_command=data[it\]
            for i in range(0,iter_time):
                res_string='ChangeColor '+ str(data['item_name'])+' '+str(data['red'])+' '+str(data['green'])+' '+str(data['blue'])+' '+str(data['alpha'])
                print(res_string)
        else:
            print('exception')