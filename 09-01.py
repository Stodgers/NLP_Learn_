from sklearn.feature_extraction.text import CountVectorizer
import jieba as jb
f = list(open('stopword.txt',encoding='utf-8'))
vec = CountVectorizer(
    analyzer='word',encoding='utf-8',stop_words=f,
    ngram_range=(1,4),max_features=100,
)

text = []
text = [" ".join(jb.lcut(i)) for i in text if i not in f]
print(text)
tt = vec.fit_transform(text)
print(tt)
