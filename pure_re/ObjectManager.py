class ObjectManager:
    def __init__(self):
        #生成不同种类的物体
        self.processer = {
            "Instantiation": self.__instantiation,
            "Delete": self.__delete,
            "Move": self.__move,
            "ChangeModel" : self.__changeModel,
            "Rotate" : self.__rotate
        }
        self.lastIndex = {
            "desk" : -1,
            "chair" : -1,
            "bed" : -1
        }
        self.lengthCount = {
            "desk" : 0,
            "chair" : 0,
            "bed" : 0
        }

    '''
    lastseqnum为一个随便的数字，不进行任何操作
    '''
    def __instantiation(self,gameobjectname,lastseqnum):
        length = self.lengthCount[gameobjectname]
        self.lengthCount[gameobjectname] = length + 1
        self.lastIndex[gameobjectname] = length + 1

    def __delete(self,gameobjectname,lastseqnum):
        pass

    def __move(self,gameobjectname,lastseqnum):
        if not self.__inrange(gameobjectname,lastseqnum):
            return
        self.lastIndex[gameobjectname] = lastseqnum

    def __changeModel(self,gameobjectname,lastseqnum):
        if not self.__inrange(gameobjectname,lastseqnum):
            return
        self.lastIndex[gameobjectname] = lastseqnum

    def __rotate(self,gameobjectname,lastseqnum):
        if not self.__inrange(gameobjectname,lastseqnum):
            return
        self.lastIndex[gameobjectname] = lastseqnum

    def __inrange(self,gameobjectname,lastseqnum):
        length = self.lengthCount[gameobjectname]
        return 1 <= lastseqnum <= length
    '''
    发送指令给客户端之前调用，用于记录指令
    如果是生成指令，lastseqnum随便给一个值即可
    instructionType指令类型
    gameobjectname物体类型
    lastseqnum本次操作的指令编号
    '''
    def recordInstruction(self,instructionType,gameobjectname,lastseqnum):
        if not gameobjectname in self.lastIndex.keys():
            return
        if not instructionType in self.processer.keys():
            return
        self.processer[instructionType](gameobjectname,lastseqnum)

    '''
    得到最后一个物体的索引
    gameobjectname物体的类型
    如果返回的是-1，代表那个不存在
    '''
    def getLastIndex(self,gameobjectname):
        if not gameobjectname in self.lastIndex.keys():
            return "exception"
        return self.lastIndex[gameobjectname]


if __name__ == "__main__":
    tmp = ObjectManager()
    tmp.recordInstruction("Instantiation","chair",-1)
    print(tmp.getLastIndex("chair"))
    #输入的物体序号不在已经生成的物体中
    tmp.recordInstruction("Delete", "chair", 3)
    print(tmp.getLastIndex("chair"))
