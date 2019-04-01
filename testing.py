import numpy as np
import scipy.stats as stats
import csv
from scipy.sparse.linalg import svds
import pandas as pd
from sklearn.preprocessing import normalize
import pickle

column = ['user_id', 'item_id', 'rating']
dataset = pd.read_csv(str('noisy_dataset.csv'), sep='\t', names=column)
# Calculating the no.of users and movies in the dataset
users = dataset.user_id.unique().shape[0]
movies = dataset.item_id.unique().shape[0]
# Building the rating matrix
matrix = np.zeros((users, movies))
i = 0
j = 0
z = 0.0
for tuple in dataset.itertuples():
    i = int(tuple[1]) - 1
    j = int(tuple[2]) - 1
    z = float(tuple[3])
    matrix[i, j] = z

column = ['user_id' , 'item_id' , 'rating' , 'timestamp']
test_dataset = pd.read_csv(str('ml-100k/u1.test') , sep = '\t' , names = column)
err = 0.0
diff = 0.0
test_data = test_dataset.iloc[: , :-1].values
x , y = test_data.shape
for i in range(0 , x):
        a = int(test_data[i][0]) - 1
        b = int(test_data[i][1]) - 1
        matrix[a][b] = 0.0

train_matrix = normalize(matrix)

"""
# get SVD components from train matrix. Choose k.
u, s, vt = svds(train_matrix, k=20)
s_diag = np.diag(s)
mat_test = np.dot(np.dot(u,s_diag) , vt)
"""
file = open('svd_mat' , 'rb')
mat_test = pickle.load(file)

column = ['user_id', 'item_id', 'rating' , 'timestamp']

# Building the rating matrix
test_matrix = np.zeros((users, movies))
i = 0
j = 0
z = 0.0
for tuple in test_dataset.itertuples():
    i = int(tuple[1]) - 1
    j = int(tuple[2]) - 1
    z = float(tuple[3])
    test_matrix[i, j] = z


test = normalize(test_matrix)

for i in range(0 , x):
        a = int(test_data[i][0]) - 1
        b = int(test_data[i][1]) - 1
        r = float(test[a][b])
        diff = np.square(r - mat_test[a][b])
        err += diff

rmse = np.sqrt(err/x)
file.close()
