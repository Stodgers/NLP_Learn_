import random
import jieba as jb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
import multiprocessing

stopwords = list(open('D:/pyy/NLP_Learn/chinese_nlp-master/data/stopwords.txt',encoding='utf-8'))
stopwords = [i[:-1] for i in stopwords]

# 加载语料
laogong_df = pd.read_csv('data/beilaogongda.csv', encoding='utf-8', sep=',')
laopo_df = pd.read_csv('data/beilaogongda.csv', encoding='utf-8', sep=',')
erzi_df = pd.read_csv('data/beierzida.csv', encoding='utf-8', sep=',')
nver_df = pd.read_csv('data/beinverda.csv', encoding='utf-8', sep=',')
# 删除语料的nan行
laogong_df.dropna(inplace=True)
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)

laogong = laogong_df.segment.values.tolist()
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()

def process_df(df,sent):
    for i in df:
        try:
            segs = jb.lcut(i)
            segs = [j for j in segs if j not in stopwords]
            segs = [j for j in segs if not str(j).isdigit()]
            segs = list(filter(lambda x:x.strip(),segs))
            segs = list(filter(lambda x:len(x)>1,segs))
            sent.append(" ".join(segs))
        except Exception:
            print(i)
            continue

sentences = []
process_df(laogong, sentences)
process_df(laopo, sentences)
process_df(erzi, sentences)
process_df(nver, sentences)
random.shuffle(sentences)
sent = sentences
for sentence in sent[:10]:
    print(sentence)

#将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
#统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(sentences))
# 获取词袋模型中的所有词语
word = vectorizer.get_feature_names()
# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()
#查看特征大小
print ('Features length: ' + str(len(word)))

print(weight)

numClass=4 #聚类分几簇
clf = KMeans(n_clusters=numClass, max_iter=10000, init="random", tol=1e-6)  #这里也可以选择随机初始化init="random"
pca = PCA(n_components=10)  # 降维
TnewData = pca.fit_transform(weight)  # 载入N维
s = clf.fit(TnewData)
print(TnewData.shape)

def plot_cluster(result, newData, numClass):
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index += 1
    color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^',
             'g^'] * 5
    for i in range(numClass):
        x1 = []
        y1 = []
        for ind1 in newData[Lab[i]]:
            # print ind1
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1, y1, color[i])

    # 绘制初始中心点
    x1 = []
    y1 = []
    for ind1 in clf.cluster_centers_:
        try:
            y1.append(ind1[1])
            x1.append(ind1[0])
        except:
            pass
    plt.plot(x1, y1, "rv")  # 绘制中心
    plt.show()

pca = PCA(n_components=2)  # 输出两维
newData = pca.fit_transform(weight)  # 载入N维
result = list(clf.predict(TnewData))
plot_cluster(result,newData,numClass)