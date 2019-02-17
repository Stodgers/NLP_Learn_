#词袋模型
import jieba as jb
from gensim import corpora
import gensim
# 定义停用词、标点符号
punctuation = ["，", "。", "：", "；", "？"]
# 定义语料
content = ["机器学习带动人工智能飞速的发展。",
           "深度学习带动人工智能飞速的发展。",
           "机器学习和深度学习带动人工智能飞速的发展。"
           ]

#获得词语并集
ll = []
seg1 = [jb.lcut(i) for i in content]
for i in seg1:
    for j in i:
        if j not in punctuation and j not in ll:
            ll.append(j)
print(ll)
#['机器', '学习', '带动', '人工智能', '飞速', '的', '发展', '深度', '和']


#去除标点
mar = []
for i in seg1:
    w = []
    for j in i:
        if j not in punctuation:
            w.append(j)
    mar.append(w)
print(mar)
#去标点符号后，我们得到结果如下：mar
# [['机器', '学习', '带动', '人工智能', '飞速', '的', '发展'],
#  ['深度', '学习', '带动', '人工智能', '飞速', '的', '发展'],
#  ['机器', '学习', '和', '深度', '学习', '带动', '人工智能', '飞速', '的', '发展']]

#将mar序列化表示为词袋向量
mar_zero = []
for i in mar:
    tk =[1 if j in i else 0 for j in ll]
    mar_zero.append(tk)
print(mar_zero)
# [[1, 1, 1, 1, 1, 1, 1, 0, 0],
#  [0, 1, 1, 1, 1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1, 1, 1, 1, 1]]

#直接获取词典
dictionary = corpora.Dictionary(mar)
print(dictionary)
print(dictionary.token2id)

cps = [dictionary.doc2bow(i) for i in mar]
print(cps)
#[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
# [(0, 1), (1, 1), (2, 1), (3, 1), (5, 1), (6, 1), (7, 1)],
# [(0, 1), (1, 1), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]