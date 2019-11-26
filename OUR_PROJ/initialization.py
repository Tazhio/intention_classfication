import os
DATA_DIR= 'data'
CLASS_PATH='Classes.txt'


if __name__=='__main__':
    # classes=[]
    class_id={
    }
    id_class={

    }
    whole_path=os.path.join(DATA_DIR,CLASS_PATH)
    with open(whole_path,'r') as f1:
        classes=f1.readlines()
    i = 1
    datasets = []
    for ele in classes:

        ele=str.strip(ele)
        class_id.update({ele:i})
        id_class.update({i:ele})

        i=i+1
        ele=ele+'.txt'
        whole_path = os.path.join(DATA_DIR,ele)
        with open(whole_path,'r') as f1:
            d1=f1.readlines()
            datasets.append(d1)



#怎么measure info extraction/slot filling 的正确性啊。



    print(class_id)
    print(id_class)
    print(datasets)





