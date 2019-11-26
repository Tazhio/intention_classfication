import argparse
from enum import Enum, unique

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sentence', type=str, default='给我一张桌子')

args = parser.parse_args()
sentence = args.sentence
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