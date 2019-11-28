import os
DATA_DIR= 'data'
CLASS_PATH='Classes.txt'


if __name__=='__main__':
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
            for line in d1:
                datasets.append(line)

    with open("dataset.csv",'w') as f2:
        for line in datasets:
            try:
                temp = line.split(" ")
                d_sentence = temp[0]
                d_class = int(temp[1].strip())
            except IndexError:
                pass
            print(d_sentence)
            print(d_class)
            new_line=d_sentence+','+id_class[d_class]+'\n'
            print(new_line)
            f2.write(new_line)







