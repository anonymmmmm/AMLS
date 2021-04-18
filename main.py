import A.A as A
import B.B as B
import pandas as pd


# ======================================================================================================================
### A Data preprocessing

A_class = A.A()
df_train_A = pd.read_table('./Datasets/SemEval2017-task4-dev.subtask-A.english.INPUT.txt', names = ['ID', 'label', 'tweet', 'NaN']) # Read training dataset
df_test_A = pd.read_table('./Datasets/SemEval2017-task4-test.subtask-A.english.txt', names = ['ID', 'label', 'tweet', 'NaN']) # Read test dataset
print('A is processing...')
X_train_A, y_train_A, embedding_layer_A, X_test_A, y_test_A = A_class.preprocess_all_in_one(df_train_A, df_test_A) # preprocess data
# ======================================================================================================================
# Task A
acc_A_train, acc_A_val, history = A_class.model_train(X_train_A, y_train_A, embedding_layer_A) # Train model based on the training set
acc_A_test = A_class.model_test(X_test_A, y_test_A)                                            # Test model based on the test set
print('A is finished')

# ======================================================================================================================
# ======================================================================================================================
### B Data preprocessing

B_class = B.B()
df_train_B = pd.read_table('./Datasets/SemEval2017-task4-dev.subtask-BD.english.INPUT.txt', names = ['ID', 'topic', 'label', 'tweet', 'NaN']) # Read training dataset
df_test_B = pd.read_table('./Datasets/SemEval2017-task4-test.subtask-BD.english.txt', names = ['ID', 'topic', 'label', 'tweet', 'NaN']) # Read test dataset
print('B is processing...')
X_train_B, y_train_B, embedding_layer_B, X_test_B, y_test_B = B_class.preprocess_all_in_one(df_train_B, df_test_B) # preprocess data
# ======================================================================================================================
# Task B
acc_B_train, acc_B_val, history = B_class.model_train(X_train_B, y_train_B, embedding_layer_B) # Train model based on the training set
acc_B_test = B_class.model_test(X_test_B, y_test_B)                                            # Test model based on the test set
print('B is finished')

print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
                                                        acc_B_train, acc_B_test))