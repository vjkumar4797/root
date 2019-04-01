import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

# Importing the dataset
rating_column = ['user_id', 'item_id', 'rating', 'timestamp']
dataset = pd.read_csv('ml-100k/u.data', sep='\t', names=rating_column)



genre_index = {"unknown" : 0 , "Action" : 1 , "Adventure" : 2, "Animation" : 3, "Children's" : 4, "Comedy" : 5, "Crime": 6 , "Documentary" : 7 , "Drama" : 8,
               "Fantasy":9 , "Film-Noir": 10 , "Horror": 11 , "Musical": 12, "Mystery": 13 , "Romance": 14 ,
               "Sci-Fi": 15, "Thriller":16 , "War":17, "Western":18}

users = 943
count = np.zeros((19,943))
genres = 19

user_genre_mat = np.zeros((users,genres))
x = 0
y = 0
i=0
j=1
ind = 0
genre_name = []
f = open('movie_genre.csv' , 'r')
file = f.readlines()

for tuple in dataset.itertuples():
    genre_name = []
    x = int(tuple[1]) - 1
    y = int(tuple[2]) - 1
    z = float(tuple[3])
    while (file[y].split('\t')[j] != '\n'):
        ind = genre_index[(file[y].split('\t')[j])]
        count[ind][x] = count[ind][x] + 1
        user_genre_mat[x][ind] += z
        j = j + 1
    j = 1

i = 0
j = 0

for i in range(0 , 943):
    for j in range(0 , 19):
        if(count[j][i] == 0):
            continue
        user_genre_mat[i][j] = round(user_genre_mat[i][j]/count[j][i] , 2)

data = user_genre_mat[0:200 , [1,14]]

# Calculating the within-cluster sum of squares
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

"""
# Determining the optimal value for the no.of clusters from the graph
no = [1,2,3,4,5,6,7,8,9,10]
plt.plot(no, wcss)
plt.title('The Elbow method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()
"""
X = data
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of users')
plt.xlabel('Action')
plt.ylabel('Romance')
plt.legend()
plt.show()