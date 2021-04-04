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

stop_words = set(stopwords.words('english'))
stop_words.remove('not')

# Read Dataset
df = pd.read_table('SemEval2017-task4-dev.subtask-A.english.INPUT.txt', header = None, names = ['ID', 'label', 'tweet', 'NaN'])
print(df)
# Data Preprocessing
def tweet_preprocess(df_tweet):
    tweet_new=[]
    for i in range(0,len(df_tweet)):
        tweet=df_tweet[i]
        # Lower
        tweet=tweet.lower()
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
print(df.tweet)

# Tokenize
tok = Tokenizer()
tok.fit_on_texts(df.tweet)
# Pad Sequence
X = pad_sequences(tok.texts_to_sequences(df.tweet), maxlen = 300)

# Build Word2Vec Model
n_dim = 300
min_count = 10
word2vec_model = gensim.models.word2vec.Word2Vec(size=n_dim, min_count=min_count)
vocabulary = [tweet.split() for tweet in df.tweet]
word2vec_model.build_vocab(vocabulary)
word2vec_model.train(vocabulary, total_examples = len(vocabulary), epochs = 100)

# Fill in Embedding Matrix
length = tok.word_index
embed_matrix = np.zeros((len(length) + 1, 300))
# word2vec_model.vector_size
####
for word, i in length.items():
    if word in word2vec_model.wv:
        embed_matrix[i] = word2vec_model.wv[word]

# Embedding Layer
embed_layer = Embedding(len(length)+1, 300, weights=[embed_matrix], input_length=300, trainable=False)

# y
y = df.label
y = pd.get_dummies(y).values

# Split train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.2, random_state = 40)

# Build, Compile and Fit LSTM Model
model = Sequential()
model.add(embed_layer)
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(196, dropout = 0.2, recurrent_dropout = 0.2)))
model.add(Dense(3, activation = 'sigmoid'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size = 256, epochs = 15, validation_split = 0.2, verbose=1)
