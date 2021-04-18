# AMLS_II_assignment20_21
Student Number: 20158088
This project solves two tasks about tweet sentiment analysis. 
Task A is Message Polarity Classification. Classify text into positive, negative, or neutral sentiment of given tweets.
Task B is Topic-Based Message Polarity Classification. Classify tweet text into positive or negative sentiment towards given topics.
In A and B folder, BiLSTM model with GloVe, BiLSTM model with Word2vec, CNN model with GloVe and Logistic Regression model with GloVe are implemented to solve task A and B. BiLSTM model with GloVe is the model selected finally for both tasks because it performs best.
In the main.py, "glove.6B.100d.txt", a pre-trained GloVe model is firstly downloaded from website. In case if it does not work, you can download this model from https://www.kaggle.com/danielwillgeorge/glove6b100dtxt.

Libraries Needed:
Wget, numpy, pandas, re, nltk, hyperopt, sklearn, tensorflow, keras, matplotlib and seaborn

In case the main function runs an error (though it should be impossible), try to change the file name.

