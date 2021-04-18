import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D

stop_words = set(stopwords.words('english'))
stop_words.remove('not')

df = pd.read_table('../Datasets/SemEval2017-task4-dev.subtask-A.english.INPUT.txt', header = None, names = ['ID', 'label', 'tweet', 'NaN'])
df_test = pd.read_table('../Datasets/SemEval2017-task4-test.subtask-A.english.txt', names = ['ID', 'label', 'tweet', 'NaN'])

def convert_number(df):
    df['label']=df['label'].replace('negative', 0)
    df['label']=df['label'].replace('neutral', 1)
    df['label']=df['label'].replace('positive', 2)
    return df
df = convert_number(df)
df_test = convert_number(df_test)

# Data Preprocessing
def tweet_preprocess(df_tweet):
    tweet_new = []
    for i in range(0,len(df_tweet)):
        tweet = df_tweet[i]
        # Lower
        tweet = tweet.lower()
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
        tweet = word_tokenize(tweet)
        tweet = [word for word in tweet if not word in stop_words]
        tweet = ' '.join(tweet)
        tweet_new.append(tweet)
    return tweet_new

df.tweet = tweet_preprocess(df.tweet)
df_test.tweet = tweet_preprocess(df_test.tweet)

# Tokenize
tok = Tokenizer()
tok.fit_on_texts(df.tweet)
vocab_size = len(tok.word_index) + 1
# Pad Sequence
X = pad_sequences(tok.texts_to_sequences(df.tweet), maxlen = 300)
X_test = pad_sequences(tok.texts_to_sequences(df_test.tweet), maxlen = 300)

embeddings_index = dict()
f = open('../glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tok.word_index.items():
    try:
        embedding_vector = embeddings_index.get(word)
    except:
        embedding_vector = embeddings_index['unknown']
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

y = to_categorical(np.asarray(df.label))
y_test = to_categorical(np.asarray(df_test.label))

X_train = X
y_train = y
embedding_layer = Embedding(input_dim = vocab_size, output_dim = 100, weights = [embedding_matrix], input_length = X_train.shape[1], trainable = False)

model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters = 100, kernel_size = 3, padding = 'valid', activation = 'relu', strides = 1))
model.add(GlobalMaxPooling1D())
model.add(Dense(3, activation = 'softmax'))
model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, validation_split = 0.2, epochs = 5, batch_size = 256)
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
loss, test_acc = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_test=np.argmax(y_test, axis=1)
y_pred=np.argmax(y_pred, axis=1)
precision=precision_score(y_test, y_pred, average = 'macro')
recall=recall_score(y_test, y_pred, average = 'macro')
f1=f1_score(y_test, y_pred, average = 'macro')
print(test_acc)
print(precision)
print(recall)
print(f1)
