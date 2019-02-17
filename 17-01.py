# -*- coding: utf-8 -*-
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
dir  = "data/"


class CorpusProcess(object):

    def __init__(self):
        """初始化"""
        self.train_process_path = dir + "train.data"  # 预处理之后的训练集
        self.test_process_path = dir + "dev.data"  # 预处理之后的测试集

    def read_corpus_from_file(self, file_path):
        """读取语料"""
        f = open(file_path, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        return lines

    def write_corpus_to_file(self, data, file_path):
        """写语料"""
        f = open(file_path, 'w')
        f.write(str(data))
        f.close()

    def process_sentence(self, lines):
        """处理句子"""
        sentence = []
        for line in lines:
            if not line.strip():
                yield sentence
                sentence = []
            else:
                lines = line.strip().split(u'\t')
                result = [line for line in lines]
                sentence.append(result)

    def initialize(self):
        """语料初始化"""
        train_lines = self.read_corpus_from_file(self.train_process_path)
        test_lines = self.read_corpus_from_file(self.test_process_path)
        self.train_sentences = [sentence for sentence in self.process_sentence(train_lines)]
        self.test_sentences = [sentence for sentence in self.process_sentence(test_lines)]

    def generator(self, train=True):
        """特征生成器"""
        if train:
            sentences = self.train_sentences
        else:
            sentences = self.test_sentences
        return self.extract_feature(sentences)

    def extract_feature(self, sentences):
        """提取特征"""
        features, tags = [], []
        for index in range(len(sentences)):
            feature_list, tag_list = [], []
            for i in range(len(sentences[index])):
                feature = {"w0": sentences[index][i][0],
                           "p0": sentences[index][i][1],
                           "w-1": sentences[index][i - 1][0] if i != 0 else "BOS",
                           "w+1": sentences[index][i + 1][0] if i != len(sentences[index]) - 1 else "EOS",
                           "p-1": sentences[index][i - 1][1] if i != 0 else "un",
                           "p+1": sentences[index][i + 1][1] if i != len(sentences[index]) - 1 else "un"}
                feature["w-1:w0"] = feature["w-1"] + feature["w0"]
                feature["w0:w+1"] = feature["w0"] + feature["w+1"]
                feature["p-1:p0"] = feature["p-1"] + feature["p0"]
                feature["p0:p+1"] = feature["p0"] + feature["p+1"]
                feature["p-1:w0"] = feature["p-1"] + feature["w0"]
                feature["w0:p+1"] = feature["w0"] + feature["p+1"]
                feature_list.append(feature)
                tag_list.append(sentences[index][i][-1])
            features.append(feature_list)
            tags.append(tag_list)
        return features, tags

class ModelParser(object):

    def __init__(self):
        """初始化参数"""
        self.algorithm = "lbfgs"
        self.c1 = 0.1
        self.c2 = 0.1
        self.max_iterations = 100
        self.model_path = "model.pkl"
        self.corpus = CorpusProcess()  #初始化CorpusProcess类
        self.corpus.initialize()  #语料预处理
        self.model = None

    def initialize_model(self):
        """模型初始化"""
        algorithm = self.algorithm
        c1 = float(self.c1)
        c2 = float(self.c2)
        max_iterations = int(self.max_iterations)
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm, c1=c1, c2=c2,
                                          max_iterations=max_iterations, all_possible_transitions=True)

    def train(self):
        """训练"""
        self.initialize_model()
        x_train, y_train = self.corpus.generator()
        self.model.fit(x_train, y_train)
        labels = list(self.model.classes_)
        x_test, y_test = self.corpus.generator(train=False)
        y_predict = self.model.predict(x_test)
        metrics.flat_f1_score(y_test, y_predict, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(metrics.flat_classification_report(y_test, y_predict, labels=sorted_labels, digits=3))
        self.save_model()

    def predict(self, sentences):
        """模型预测"""
        self.load_model()
        features, _ = self.corpus.extract_feature(sentences)
        return self.model.predict(features)

    def load_model(self, name='model'):
        """加载模型 """
        self.model = joblib.load(self.model_path)

    def save_model(self, name='model'):
        """保存模型"""
        joblib.dump(self.model, self.model_path)

model = ModelParser()
model.train()

sen =[[['坚决', 'a', 'ad', '1_v'],
  ['惩治', 'v', 'v', '0_Root'],
  ['贪污', 'v', 'v', '1_v'],
  ['贿赂', 'n', 'n', '-1_v'],
  ['等', 'u', 'udeng', '-1_v'],
  ['经济', 'n', 'n', '1_v'],
  ['犯罪', 'v', 'vn', '-2_v']]]
print(model.predict(sen))