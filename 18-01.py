from sklearn.externals import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding, GRU
from keras.models import Sequential
from flask import Flask,request
app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/predict/<sen>')
def predict(sen):
    result = model.predict(sen)
    return str(result)
if __name__ == '__main__':
    model = joblib.load('lstm.pkl')
    app.run(host='0.0.0.0',port=8088)