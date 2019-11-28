import argparse
import sys
from INTENTS.Instantiation import Instantiation
from INTENTS.Delete1 import Delete
from use_classifier import intent_classifier
from INTENTS.Transition1 import Transition1
from INTENTS.ChangeColor1 import ChangeColor1


#
# def returnITEMNUM():
#     return ITEM_NUM

if __name__ == '__main__':
    ITEM_NUM = 0
    user_sentence = sys.argv[1]
    intent=intent_classifier(user_sentence)
    # try:
    if(intent==1):
        instant = Instantiation()
        instant.analyze(user_sentence)
        ITEM_NUM= ITEM_NUM + 1
    if(intent==2):
        ITEM_NUM = ITEM_NUM - 1
        # assert ITEM_NUM >= 0
        delete1=Delete()
        delete1.analyze(user_sentence)

    if(intent==3):
        transition=Transition1()
        transition.analyze(user_sentence)

    if(intent==4):
        changecolor1=ChangeColor1()
        changecolor1.analyze(user_sentence)



    # except AssertionError:
    #     print("There is no more item to delete")


