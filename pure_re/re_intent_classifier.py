import re




re_collection=[
]
color=["亮", "暗","色","颜","调.",'变.*色']
instantiation=["拿","整","给","找","需要"]
delete=["删","丢","去掉","滚","碍眼","不想要"]
move=["挪","移","抬"]
def re_classifier(user_sentence):
    for expressions in color:
        match_result = re.search(expressions, user_sentence)
        if not match_result is None:
            return 4
    for expressions in delete:
        match_result = re.search(expressions, user_sentence)
        if not match_result is None:
            return 2
    for expressions in instantiation:
        match_result = re.search(expressions, user_sentence)
        if not match_result is None:
            return 1
    for expressions in move:
        match_result = re.search(expressions, user_sentence)
        if not match_result is None:
            return 3












