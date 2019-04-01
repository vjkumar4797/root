import numpy as np
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



def clustering(rating_matrix):

    """
    #Calculating the within-cluster sum of squares
    wcss = []


    for i in range(1,25):
        kmeans = KMeans(n_clusters = i , init = 'k-means++'  )
        kmeans.fit(rating_matrix)
        wcss.append(kmeans.inertia_)


    #Determining the optimal value for the no.of clusters from the graph
    plt.plot(range(1,25), wcss)
    plt.title('The Elbow method')
    plt.xlabel('No. of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    """

    #Fitting k-means to the dataset
    kmeans = KMeans(n_clusters=5, init='k-means++')
    kmeans.fit_predict(rating_matrix)

    #Centre - Item rating matrix

    centres = np.array(kmeans.cluster_centers_)

    #Appending labels to the user-item rating matrix
    labels = np.array(kmeans.labels_)
    A_matrix = np.c_[rating_matrix, labels]

    return A_matrix , centres



