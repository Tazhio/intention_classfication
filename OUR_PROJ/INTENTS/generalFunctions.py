

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




def CHINESE_STRING_TO_INT(chinese_str:str):
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

