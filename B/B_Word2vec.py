import numpy as np
import pandas as pd
import re
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

stop_words = set(stopwords.words('english'))
stop_words.remove('not')

df = pd.read_table('../Datasets/SemEval2017-task4-dev.subtask-BD.english.INPUT.txt', names = ['ID', 'topic', 'label', 'tweet', 'NaN'])
df_test = pd.read_table('../Datasets/SemEval2017-task4-test.subtask-BD.english.txt', names = ['ID', 'topic', 'label', 'tweet', 'NaN'])
df['label'] = df['label'].replace('negative', 0)
df['label'] = df['label'].replace('positive', 1)

df_test['label'] = df_test['label'].replace('negative', 0)
df_test['label'] = df_test['label'].replace('positive', 1)

# Data Preprocessing
def tweet_preprocess(df_tweet):
    tweet_new=[]
    for i in range(0,len(df_tweet)):
        tweet=df_tweet[i]
        # Lower
        tweet=tweet.lower()
        # Remove emoji
        emoji = re.compile("["
                         u"\U0001F600-\U0001F64F"  # emoticons
                         u"\U0001F300-\U0001F5FF"  # pictographs, symbols
                         u"\U0001F680-\U0001F6FF"  # map symbols, transport
                         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                         u"\U00002702-\U000027B0"
                         u"\U000024C2-\U0001F251"
                         "]+", flags=re.UNICODE)
        tweet = emoji.sub(r'', tweet)
        # Remove urls
        tweet = re.sub(r"http\S+|www\S+|https\S+", ' ', tweet)
        # Remove @user and #hashtag
        tweet = re.sub(r'\@\w+|\#', ' ', tweet)
        # Remove special numbers and punctuations
        tweet = re.sub(r'[^A-Za-z0-9]', ' ', tweet)
        # Remove stop words
        tweet=word_tokenize(tweet)
        tweet=[word for word in tweet if not word in stop_words]
        tweet=' '.join(tweet)
        tweet_new.append(tweet)
    return tweet_new

df.tweet = tweet_preprocess(df.tweet)
df_test.tweet = tweet_preprocess(df_test.tweet)

# Tokenize
tok = Tokenizer()
tok.fit_on_texts(df.tweet)
# Pad Sequence
X = pad_sequences(tok.texts_to_sequences(df.tweet), maxlen = 100)
X_test = pad_sequences(tok.texts_to_sequences(df_test.tweet), maxlen = 100)

# Build Word2Vec Model
n_dim = 100
min_count = 10
word2vec_model = gensim.models.word2vec.Word2Vec(size = n_dim, min_count = min_count)
vocabulary = [tweet.split() for tweet in df.tweet]
word2vec_model.build_vocab(vocabulary)
word2vec_model.train(vocabulary, total_examples = len(vocabulary), epochs = 100)

length = tok.word_index
embed_matrix = np.zeros((len(length) + 1, 100))
for word, i in length.items():
    if word in word2vec_model.wv:
        embed_matrix[i] = word2vec_model.wv[word]

embed_layer = Embedding(len(length)+1, 100, weights=[embed_matrix], input_length=100, trainable=False)

y = to_categorical(np.asarray(df.label))
y_test = to_categorical(np.asarray(df_test.label))

X_train = X
y_train = y

model = Sequential()
model.add(embed_layer)
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(128, dropout = 0.2, recurrent_dropout = 0.5)))
model.add(Dense(2, activation = 'softmax'))
model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, validation_split = 0.2, epochs = 15, batch_size = 256)
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
loss, test_acc = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_test=np.argmax(y_test, axis = 1)
y_pred=np.argmax(y_pred, axis = 1)
precision=precision_score(y_test, y_pred, average = 'macro')
recall=recall_score(y_test, y_pred, average = 'macro')
f1=f1_score(y_test, y_pred, average = 'macro')
print(test_acc)
print(precision)
print(recall)
print(f1)
