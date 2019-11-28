```
    #先写，先实现功能，后续再想怎么修改和怎么优化可维护性吧。
    #先暂时这样，好像有不少细节要想啊，要有整个体系要架构，感觉还挺爽。





#pseduo-code:
# 用栈实现转化：
# 如果新读入的汉字(假设它代表的数字为N)
# 比栈顶数字还要大
# {
#     弹出栈中所有比N小的元素，并将这些元素累加，假设结果为S;
# 将S * N入栈；
# }
# 否则
# {
#     将N直接入栈;
# }
# 最后将栈中所有数字相加。
#https://blog.csdn.net/taoqick/article/details/40819095
```





Back up codes





```
# 测试文本
test = '<h1>hello 你好, world 世界</h1>'

# 中文匹配正则
chinese_pattern = '[\u4e00-\u9fa5]+'
says = re.findall(chinese_pattern, test)
# 输出提取的内容
hi = ''
for say in says:
    # print(say)
    hi += say + ','
hi = hi.strip(',')

# 打印结果：你好,世界
print(hi)

str2 = '我需要十一个桌子 1'
res = re.search('.*需要(?P<quantity>[零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬1234567890]+)个(?P<item_name>.*)', str2)
print(res.groupdict())

str3 = '椅子在哪里'
res2 = re.search('(?P<item_name>.*)在哪里', str3)
print(res2.groupdict())

s = '1102231990xxxxxxxx'
res = re.search('(?P<province>\d{3})(?P<city>\d{3})(?P<born_year>\d{4})', s)
print(res.groupdict())
ress=Chinese_String_to_int("一百一十二")
print(ress)
```