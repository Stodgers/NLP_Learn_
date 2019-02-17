import jieba
import jieba.analyse
import jieba.posseg as posg

sentence = u'''上线三年就成功上市,拼多多上演了互联网企业的上市奇迹,却也放大平台上存在的诸多问题，拼多多在美国上市。'''
kw = jieba.analyse.extract_tags(sentence, topK=10, withWeight=True, allowPOS=('n', 'ns'))
for item in kw:
    print(item[0], item[1])


kw=jieba.analyse.textrank(sentence,topK=20,withWeight=True,allowPOS=('ns','n'))
for item in kw:
    print(item[0],item[1])

#pip install pyhanlp
from pyhanlp import *

sentence = u'''上线三年就成功上市,拼多多上演了互联网企业的上市奇迹,却也放大平台上存在的诸多问题，拼多多在美国上市。'''
analyzer = PerceptronLexicalAnalyzer()
segs = analyzer.analyze(sentence)
arr = str(segs).split(" ")


def get_result(arr):
    re_list = []
    ner = ['n', 'ns']
    for x in arr:
        temp = x.split("/")
        if (temp[1] in ner):
            re_list.append(temp[0])
    return re_list

result = get_result(arr)
print(result)