import numpy as np
import scipy.stats as stats
import csv
from scipy.sparse.linalg import svds
import pandas as pd

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

#User Number
user = 20

m_header = []
movies = open('ml-100k/u.item')
data = movies.readlines()
for f in data:
    m_header.append(f.split('|')[1])

# get SVD components from train matrix. Choose k.
u, s, vt = svds(matrix, k=20)
s_diag = np.diag(s)
mat = np.dot(np.dot(u[user], s_diag), vt)

# No.of recommendations
n = 5
max = np.argpartition(mat, -n)[-n:]
sugg = []
index = []
for ind in max:
    index.append(ind)
    sugg.append(m_header[ind])


i = 0
j = 0
rec_list = []
f = open('movie_genre.csv', 'r')
file = f.readlines()
print("User No:", user)
for i in index:
    rec_list = []
    while (file[i].split('\t')[j] != '\n'):
        rec_list.append(file[i].split('\t')[j])
        j = j + 1
    j = 0
    print(rec_list)
