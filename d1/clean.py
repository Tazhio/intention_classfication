import re

if __name__=="__main__":
    re_collection=["桌"]



    with open("fengrufeitun_moyan.txt","r") as f1:
        str1=f1.read()

    str1=str1.replace(';','。')
    str1=str1.replace('，', '。')
    str1=str1.replace('；', '。')
    str1 = str1.replace('\r', '。')
    str1 = str1.replace('\n', '。')
    # str1=str1.replace('\n\r', '。')
    str1 = str1.replace('？', '。')
    str1 = str1.replace('！', '。')
    str1 = str1.replace('“', '。')
    str1 = str1.replace('”', '。')




    sps1=str1.split("。")
    # print(sps1)
    # for ele in sps1:
    #     print(ele)

    ress=[]

    for ele in sps1:
        match_result = re.search("桌", ele)
        if not match_result is None:
            ress.append(ele)
            print(ele)

    print(ress)


