import pandas as pd
import jieba as jb
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
df = pd.read_csv('datascience.csv',encoding='gb18030')
#print(df.head())
#print(df.shape)
f = list(open('stopword.txt',encoding='utf-8'))
n_features = 1000
def word_cut(text):
    return " ".join(jb.cut(text))


df["content_cutted"] = df.content.apply(word_cut)

print(df.loc[0:4,['content','content_cutted']])
n_features = 1000
#strip_accents 默认为None，可设为ascii或unicode，将使用ascii或unicode编码在预处理步骤去除raw document中的重音符号
#max_features
#stop_words 停用词，设为english将使用内置的英语停用词，设为一个list可自定义停用词，设为None不使用停用词，
#           设为None且max_df∈[0.7, 1.0)将自动根据当前的语料库建立停用词表
#max_df 可以设置为范围在[0.0 1.0]的float，也可以设置为没有范围限制的int，默认为1.0。
#       这个参数的作用是作为一个阈值，当构造语料库的关键词集的时候，
#       如果某个词的document frequence大于max_df，这个词不会被当作关键词。
#min_df 类似于max_df，不同之处在于如果某个词的document frequence小于min_df，则这个词不会被当作关键词
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words=f,
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(df.content_cutted)

#https://www.cnblogs.com/pinard/p/6908150.html
#n_topics: 即我们的隐含主题数K
#max_iter ：EM算法的最大迭代次数。
#learning_method: 即LDA的求解算法。有 ‘batch’ 和 ‘online’两种选择。
#                  ‘batch’即我们在原理篇讲的变分推断EM算法，而"online"即在线变分推断EM算法，
#                  在"batch"的基础上引入了分步训练，将训练样本分批，逐步一批批的用样本更新主题词分布的算法。
#                  默认是"online"。选择了‘online’则我们可以在训练时使用partial_fit函数分布训练。
#                  不过在scikit-learn 0.20版本中默认算法会改回到"batch"。
#                  建议样本量不大只是用来学习的话用"batch"比较好，这样可以少很多参数要调,
#                  而样本太多太大的话，"online"则是首先了。
n_topics = 10
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(tf)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    #print()

n_top_words = 20
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
pyLDAvis.show(data)