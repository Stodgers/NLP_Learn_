import pandas as pd
import jieba as jb
import jieba as jb
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
df = pd.read_csv('datascience.csv',encoding='gb18030')

ll = jb.lcut('老哥们请教个事情，我有个z87主板，3条pcie的，16x3速度  支持cf和sli，我能双卡cf然后第三个用pcie转接M2达到满速吗')
tt = [i for i in ll if i!='，' and i!=' ']
print("/".join(tt))

st = open('stopword.txt',encoding='utf-8')

sl = [i for i in ll if i not in st]
print(sl)

Lab = [[] for i in range(5)]
print(Lab)