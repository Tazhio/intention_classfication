import argparse
from enum import Enum,unique
import re


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

def info_extraction_re(text:str,intent:int):







    return

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








