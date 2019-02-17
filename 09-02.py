import random
import numpy as np
import jieba as jb
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding, GRU
from keras.models import Sequential
from keras.optimizers import SGD
stopwords = open('D:/pyy/NLP_Learn/chinese_nlp-master/data/stopwords.txt',encoding='utf-8')

laogong_df = pd.read_csv('data/beilaogongda.csv', encoding='utf-8', sep=',')
laopo_df = pd.read_csv('data/beilaogongda.csv', encoding='utf-8', sep=',')
erzi_df = pd.read_csv('data/beierzida.csv', encoding='utf-8', sep=',')
nver_df = pd.read_csv('data/beinverda.csv', encoding='utf-8', sep=',')
laogong_df.dropna(inplace=True)
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)
laogong = laogong_df.segment.values.tolist()
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()

def process_text(ll,sent,tag):
    for i in ll:
        try:
            segs = jb.lcut(i)
            segs = [v for v in segs if not str(v).isdigit()]
            segs = list(filter(lambda x: x.strip(), segs))
            segs = list(filter(lambda x: len(x) > 1, segs))
            segs = list(filter(lambda x: x not in stopwords, segs))
            sent.append((" ".join(segs),tag))
        except Exception:
            print(i)
            continue

sent = []
process_text(laogong, sent,0)
process_text(laopo, sent, 1)
process_text(erzi, sent, 2)
process_text(nver, sent, 3)
random.shuffle(sent)
print(sent[:10])

all_texts = [i[0] for i in sent]
all_labels = [i[1] for i in sent]
print((all_texts[:10]))
print((all_labels[:10]))

# config
MAX_SEQUENCE_LENGTH = 100  # 最大序列长度
EMBEDDING_DIM = 200  # embdding 维度
VALIDATION_SPLIT = 0.16  # 验证集比例
TEST_SPLIT = 0.2  # 测试集比例

#keras的sequence模块文本序列填充
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


p1 = int((1-0.16)*len(data))
p2 = int(0.8*len(data))

x_train = data[:p1]
y_train = labels[:p1]

x_val = data[p1:p2]
y_val = labels[p1:p2]

x_test = data[p2:]
y_test = labels[p2:]

model = Sequential()
model.add(Embedding(len(word_index)+1,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(64, activation='tanh'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  #optimizer='sgd',
                  metrics=['acc'])
print(model.metrics_names)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)
model.save('lstm.h5')
# 模型评估
print(model.evaluate(x_test, y_test))