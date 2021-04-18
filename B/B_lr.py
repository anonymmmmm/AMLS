import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import pickle
import re
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

stop_words = set(stopwords.words('english'))
stop_words.remove('not')

df = pd.read_table('../Datasets/SemEval2017-task4-dev.subtask-BD.english.INPUT.txt', names = ['ID', 'topic', 'label', 'tweet', 'NaN'])
df1 = pd.read_table('../Datasets/SemEval2017-task4-test.subtask-BD.english.txt', names = ['ID', 'topic', 'label', 'tweet', 'NaN'])
df['label'] = df['label'].replace('negative', 0)
df['label'] = df['label'].replace('positive', 1)

df1['label'] = df1['label'].replace('negative', 0)
df1['label'] = df1['label'].replace('positive', 1)

X = df['tweet']
y = df['label']

X1 = df1['tweet']
y1 = df1['label']

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
        tweet = re.sub(r'http\S+|www\S+|https\S+', ' ', tweet)
        # Remove @user and #hashtag
        tweet = re.sub(r'\@\w+|\#', ' ', tweet)
        # Remove special numbers and punctuations
        tweet = re.sub(r'[^A-Za-z0-9]', ' ', tweet)
        # Remove numbers again
        tweet = re.sub(r'[0-9]', ' ', tweet)
        # Remove short words which have length 3 or less
        tweet = ' '.join([word for word in tweet.split() if len(word) > 2])
        # Remove stop words
        tweet = word_tokenize(tweet)
        tweet = [word for word in tweet if not word in stop_words]
        tweet = ' '.join(tweet)
        tweet_new.append(tweet)
    return tweet_new

X = tweet_preprocess(df.tweet)
X1 = tweet_preprocess(df1.tweet)

# Tokenize
tok = Tokenizer()
tok.fit_on_texts(df.tweet)
vocab_size = len(tok.word_index) + 1
# Pad Sequence
X_train = pad_sequences(tok.texts_to_sequences(X), maxlen = 100)
X_test = pad_sequences(tok.texts_to_sequences(X1), maxlen = 100)

y_train = y
y1 = y1

embeddings_index = dict()
f = open('../glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += embeddings_index[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
wordvec = np.zeros((len(X_train), 100))
for i in range(len(X_train)):
    wordvec[i,:] = word_vector(X_train[i], 100)
wordvec_df = pd.DataFrame(wordvec)

wordvec_test = np.zeros((len(X1), 100))
for i in range(len(X1)):
    wordvec_test[i,:] = word_vector(X1[i], 100)
wordvec_df_test = pd.DataFrame(wordvec_test)

X_train, X_valid, y_train, y_valid = train_test_split( wordvec_df, y_train, test_size = 0.2, random_state = 5)
lr = LogisticRegression(solver = 'lbfgs')
lr.fit(X_train, y_train)
prediction = lr.predict(wordvec_df_test)
score = lr.score(X_valid, y_valid)
score1 = lr.score(wordvec_df_test, y1)
precision = precision_score(y1, prediction, average = 'macro')
recall = recall_score(y1, prediction, average = 'macro')
f1 = f1_score(y1, prediction, average = 'macro')
print(score)
print(score1)
print(precision)
print(recall)
print(f1)