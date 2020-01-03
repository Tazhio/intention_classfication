def reading_files(path):
    with open(path, 'r') as f1:
        file1 = f1.readlines()
        res_list = []
        for ele in file1:
            res_list.append(ele.strip())
    print(res_list)
    return res_list

if __name__=='__main__':


    #reading data files
    colors=reading_files('color.txt')
    objects=reading_files('object_names.txt')
    directions=reading_files('direction.txt')
    direction_words=reading_files('direction_words.txt')
    replace_verbs=reading_files('replace.txt')
    transition_verbs=reading_files('transition.txt')
    delete_verbs=reading_files('delete.txt')
    initialization_verbs=reading_files('initialization.txt')
    verbs=reading_files('verbs.txt')


    res_list=[]


    for obj in objects:
        for veb in verbs:
            for dir in directions:
                for dirw in direction_words:
                    print(obj+veb+dir+dirw)
                    res_list.append(obj+veb+dir+dirw+'\n')
    print(len(res_list))

    with open('new_data.txt','w') as f1:
        f1.writelines(res_list)
