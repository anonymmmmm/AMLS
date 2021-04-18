import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SpatialDropout1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class B:
    # Read dataset
    def convert_number(self, df):
        """
        Convert labels(negative, positive) to 0, 1
        :param df: The dataset just read in
        :return: The dataset after converting
        """
        df['label'] = df['label'].replace('negative', 0)
        df['label'] = df['label'].replace('positive', 1)
        return df

    def topic_preprocess(self, df):
        """
        If topic is not in the corresponding tweet text, add it in
        :param df: dataset
        :return: dataset
        """
        df_tweet = df.tweet
        df_topic = df.topic
        for i in range(0, len(df_tweet)):
            if df_tweet[i].find(df_topic[i])==-1:
                df_tweet[i]=df_topic[i]+' '+df_tweet[i]
        df.tweet = df_tweet
        return df

    def tweet_preprocess(self, df_tweet):
        """
        Preprocess tweet text
        :param df_tweet: The tweet text df['tweet']
        :return: The tweet text after preprocessing df['tweet']
        """
        tweet_new=[]
        for i in range(0,len(df_tweet)):
            tweet=df_tweet[i]
            # Lower
            tweet=tweet.lower()
            # Remove emoji
            emoji=re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # pictographs, symbols
                             u"\U0001F680-\U0001F6FF"  # map symbols, transport
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)
            tweet=emoji.sub(r'', tweet)
            # Remove urls
            tweet = re.sub(r'http\S+|www\S+|https\S+', ' ', tweet)
            # Remove @user and #hashtag
            tweet = re.sub(r'\@\w+|\#', ' ', tweet)
            # Remove special numbers and punctuations
            tweet = re.sub(r'[^A-Za-z0-9]', ' ', tweet)
            # Remove numbers again
            tweet = re.sub(r'[0-9]', ' ', tweet)
            # Remove short words which have length 3 or less
            tweet = ' '.join([word for word in tweet.split() if len(word) > 3])
            # Remove stop words
            # stop words
            stop_words = set(stopwords.words('english'))
            stop_words.remove('not')
            tweet = word_tokenize(tweet)
            tweet = [word for word in tweet if not word in stop_words]
            tweet = ' '.join(tweet)
            tweet_new.append(tweet)
        return tweet_new

    # Vectorize texts
    def tokenize_train(self, df_tweet):
        """
        Vectorize texts
        :param df_tweet: The tweet text df['tweet']
        :return: Tweet texts after vectorizing, vocabulary size
        """
        tok = Tokenizer()
        # Create vocabulary index based on word frequency
        tok.fit_on_texts(df_tweet)
        # Convert each text to a sequence of integers
        X = pad_sequences(tok.texts_to_sequences(df_tweet), maxlen=100)
        # Vocabulary size
        vocab_size = len(tok.word_index) + 1
        return X, vocab_size, tok

    # Vectorize texts
    def tokenize_test(self, tok, df_tweet):
        """
        Victorize texts
        :param df_tweet: The tweet text df['tweet']
        :return: Tweet texts after vectorizing, vocabulary size
        """
        # Convert each text to a sequence of integers
        X = pad_sequences(tok.texts_to_sequences(df_tweet), maxlen=100)
        return X

    def one_hot(self, df_label):
        """
        Convert sentiment labels to one-hot encode
        :param df_label: sentiment label
        :return: sentiment label
        """
        y = to_categorical(np.asarray(df_label))
        return y

    # Shuffle train data
    def shuffle_train(self, X_train, y_train):
        """
        Shuffle train data
        :param X_train: tweet train data
        :param y_train: sentiment label train data
        :return: tweet train data, sentiment label train data
        """
        state = np.random.get_state()
        np.random.shuffle(X_train)
        np.random.set_state(state)
        np.random.shuffle(y_train)
        return X_train, y_train

    def glove_load(self):
        """
        Load pre-trained GloVe model
        :return: embedding index
        """
        embeddings_index = dict()
        f = open('glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def build_matrix_and_layer(self, vocab_size, tok, embeddings_index):
        """
        Build embedding matrix and embedding layer
        :param vocab_size: vocabulary size
        :param tok: tokenizer
        :param embeddings_index: embedding index
        :return: embedding matrix and embedding layer
        """
        #Build embedding matrix
        embedding_matrix = np.zeros((vocab_size, 100))
        for word, i in tok.word_index.items():
            try:
                # Vector corresponds to word
                embedding_vector = embeddings_index.get(word)
            except:
                embedding_vector = embeddings_index['unknown']
            if embedding_vector is not None:
                # Ensure vector of embedding_matrix row matches word index
                embedding_matrix[i] = embedding_vector
        # Build embedding layer
        embedding_layer = Embedding(input_dim = vocab_size, output_dim = 100, weights = [embedding_matrix], input_length = 100, trainable=False)
        return embedding_layer

    def model_train(self, X_train, y_train, embedding_layer):
        """
        Train, validate and test BiLSTM model, calculate accuracy of training and validation set
        :param X_train: tweet train data
        :param y_train: sentiment label train data
        :param embedding_layer: embedding layer
        :param X_test: tweet test data
        :param y_test: sentiment label test data
        :return: accuracy, recall, precision, F1 score and history
        """
        model = Sequential()
        model.add(embedding_layer)
        model.add(SpatialDropout1D(0.3))
        model.add(Bidirectional(LSTM(64, dropout = 0.1, recurrent_dropout = 0.3)))
        model.add(Dense(2, activation = 'softmax'))
        model.summary()
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history=model.fit(X_train, y_train, validation_split = 0.2, epochs = 21, batch_size = 128)
        model.save('taskB.h5')
        train_acc=history.history['accuracy'][-1]
        val_acc=history.history['val_accuracy'][-1]
        return train_acc, val_acc, history

    def model_test(self, X_test, y_test):
        """
        Test BiLSTM model
        :param X_test: tweet test dataset
        :param y_test: sentiment test dataset
        :return: test accuracy
        """
        model=load_model('taskB.h5')
        loss, test_acc=model.evaluate(X_test, y_test)
        return test_acc

    def preprocess_all_in_one(self, df_train, df_test):
        """
        Integrated data processing
        :param df_train: train data
        :param df_test: test data
        :return:
        """
        df_train = self.convert_number(df_train)
        X = self.tweet_preprocess(df_train.tweet)
        X, vocab_size, tok = self.tokenize_train(X)
        y = self.one_hot(df_train.label)
        X_train, y_train=self.shuffle_train(X, y)
        embeddings_index=self.glove_load()
        embedding_layer=self.build_matrix_and_layer(vocab_size, tok, embeddings_index)

        df_test = self.convert_number(df_test)
        X_test = self.tweet_preprocess(df_test.tweet)
        X_test = self.tokenize_test(tok, X_test)
        y_test = self.one_hot(df_test.label)

        return X_train, y_train, embedding_layer, X_test, y_test

    def learning_curve(self, history):
        """
        Plot learning curve about accuracy, and loss
        """
        # accuracy over epochs
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Learning Curve Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        # loss over epochs
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Learning Curve Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def extra(self, y_test, y_pred):
        """
        Plot confusion matrix, calculate precision, recall and F1 score of test set
        """
        f, ax = plt.subplots()
        y_test = np.argmax(y_test, axis = 1)
        y_pred = np.argmax(y_pred, axis = 1)
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, ax=ax)
        ax.set_xlabel('Predict')
        ax.set_ylabel('True')
        plt.show()

        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        return precision, recall, f1

    def hyperparameter(self, params, embedding_layer, X_train, y_train, X_test, y_test):
        """
        Tune hyper-parameters
        """
        DROPOUT_CHOICES = np.arange(0.0, 1, 0.1)
        UNIT_CHOICES = np.arange(32, 128, 32, dtype = int)
        BATCH_UNITS = np.arange(64, 256, 64, dtype = int)
        space={
            'units': hp.choice('units', UNIT_CHOICES),
            'batch_size': hp.choice('batch_size', BATCH_UNITS),
            'dropout1': hp.choice('dropout1', DROPOUT_CHOICES),
            'dropout2': hp.choice('dropout2', DROPOUT_CHOICES),
            'dropout3': hp.choice('dropout2', DROPOUT_CHOICES),
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop'])
        }
        print('Params testing: ', params)
        model=Sequential()
        model.add(embedding_layer)
        model.add(SpatialDropout1D(params['dropout1']))
        model.add(Bidirectional(LSTM(params['units'], dropout = params['dropout2'], recurrent_dropout = params['dropout3'])))
        model.add(Dense(3, activation = 'softmax'))
        model.summary()
        model.compile(optimizer = params['optimizer'], loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history = model.fit(X_train, y_train, validation_split = 0.2, epochs = 21, batch_size = params['batch_size'])
        pred = model.predict(X_test, verbose=0)
        acc = (pred.argmax(axis = 1)==y_test.argmax(axis = 1)).mean()
        trials = Trials()
        best = fmin(model, space, algo = tpe.suggest, max_evals=20, trials = trials)
        print('best parameters: ', best)

