# 语料加载
# 分词
# 去停用词
# 抽取词向量特征
# 分别进行算法建模和模型训练
# 评估、计算 AUC 值
# 模型对比
import random
import jieba as jb
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import numpy as np


stopwords = pd.read_csv('D:/pyy/NLP_Learn/chinese_nlp-master/data/stopwords.txt',
                        index_col=False,
                        quoting=3,
                        sep="\t",
                        names=['stopword'],
                        encoding='utf-8'
                        )
stopwords = stopwords['stopword'].values
print(list(stopwords))

#加载语料数据
laogong_df = pd.read_csv('data/beilaogongda.csv', encoding='utf-8', sep=',')
laopo_df = pd.read_csv('data/beilaogongda.csv', encoding='utf-8', sep=',')
erzi_df = pd.read_csv('data/beierzida.csv', encoding='utf-8', sep=',')
nver_df = pd.read_csv('data/beinverda.csv', encoding='utf-8', sep=',')

#删除语料的nan行
laogong_df.dropna(inplace=True)
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)

#转换
laogong = laogong_df.segment.values.tolist()
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()

print(laogong)



def process_text(ll,sent,catgory):
    for i in ll:
        try:
            segs = jb.lcut(i)
            segs = [v for v in segs if not str(v).isdigit()]  # 去数字
            segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
            segs = list(filter(lambda x: len(x) > 1, segs))  # 长度为1的字符
            segs = list(filter(lambda x: x not in stopwords, segs))  # 去掉停用词
            sent.append((" ".join(segs), catgory))  # 打标签
            '''
            segs = jb.lcut(i)
            segs = [j for j in segs if j not in stopwords]
            segs = [j for j in segs if not str(j).isdigit()]
            segs = list(filter(lambda x:x.strip(),segs))
            segs = list(filter(lambda x:len(x)>1,segs))
            sent.append((" ".join(segs),catgory))
            '''
        except Exception:
            print(i)
            continue

sent = []
process_text(laogong, sent, 0)
process_text(laopo, sent, 1)
process_text(erzi, sent, 2)
process_text(nver, sent, 3)

#print(sent)

random.shuffle(sent)

for i in sent[:10]:
    print(i[0],i[1])



vec = CountVectorizer(analyzer='word',ngram_range=(1,4),max_features=20000)
x, y = zip(*sent)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1256)
# x被划分的样本集
# y被划分的样本标签
# random_state随机种子

#把训练数据转换为词袋模型：
vec.fit(x_train)


#xgboost
xgb_train = xgb.DMatrix(vec.transform(x_train), label=y_train)
xgb_test = xgb.DMatrix(vec.transform(x_test))
params = {
            'booster': 'gbtree',     #使用gbtree
            'objective': 'multi:softmax',  # 多分类的问题、
            # 'objective': 'multi:softprob',   # 多分类概率
            #'objective': 'binary:logistic',  #二分类
            'eval_metric': 'merror',   #logloss
            'num_class': 4,  # 类别数，与 multisoftmax 并用
            'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth': 8,  # 构建树的深度，越大越容易过拟合
            'alpha': 0,   # L1正则化系数
            'lambda': 10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample': 0.7,  # 随机采样训练样本
            'colsample_bytree': 0.5,  # 生成树时进行的列采样
            'min_child_weight': 3,
            # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            # 假设 h 在 0.01 附近，min_child_weight 为 1 叶子节点中最少需要包含 100 个样本。
            'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
            'eta': 0.03,  # 如同学习率
            'seed': 1000,
            'nthread': -1,  # cpu 线程数
            'missing': 1
        }
plst = list(params.items())
num_rounds = 200  # 迭代次数
watchlist = [(xgb_train, 'train')]
# 交叉验证
#result = xgb.cv(plst, xgb_train, num_boost_round=200, nfold=4, early_stopping_rounds=200, verbose_eval=True, folds=StratifiedKFold(n_splits=4).split(vec.transform(x_train), y_train))
# 训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=200)
#model.save_model('../data/model/xgb.model')  # 用于存储训练出的模型
ans = model.predict(xgb_test)   #预测

k1 = 0.0
k2 = 0.0
for i in range(len(ans)):
    if(ans[i]==y_test[i]):k1+=1
    k2+=1
print("XGboost: ",k1/k2)

#Naive Bayesian
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)
print("Naive Bayesian: ",classifier.score(vec.transform(x_test), y_test))

#svm
svm = SVC(kernel='linear')
svm.fit(vec.transform(x_train), y_train)
print("SVM: ",svm.score(vec.transform(x_test), y_test))

'''
XGboost:  0.6682134570765661
Naive Bayesian:  0.642691415313225
SVM:  0.6264501160092807
'''