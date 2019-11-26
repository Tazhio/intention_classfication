
class config:
    def __init__(self):

        self.loc_x=0
        self.loc_y=0
        self.item_type=0
        self.item_id=0
        self.off_setx=0
        self.off_sety=0
        self.rgb_red = 0
        self.rgb_blue = 0
        self.rgb_green=0


        return

    def produre_file(self):
        to_write=[]
        to_write.append('x:'+str(self.loc_x))
        to_write.append('y:'+str(self.loc_y))
        to_write.append('item_id'+str(self.item_id))
        to_write.append('item_type:'+str(self.item_type))

        to_write.append('off_setx:' + str(self.off_setx))
        to_write.append('off_sety:' + str(self.off_sety))
        to_write.append('red:' + str(self.rgb_red))
        to_write.append('blue' + str(self.rgb_blue))
        to_write.append('green' + str(self.rgb_green))


        with open('user_operation.txt','w')as f1:
            for lin in to_write:
                f1.write(lin+'\r\n')
            # f1.writelines(to_write)




if __name__=='__main__':
    config1=config()

    config1.produre_file()

    print('write finished')

