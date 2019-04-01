import numpy as np
import pandas as pd


def pre_processing(filename):

    #Importing the dataset
    rating_column = ['user_id', 'item_id', 'rating', 'timestamp']
    dataset = pd.read_csv(filename , sep='\t' , names = rating_column)

    #Calculating the no.of users and movies in the dataset
    users = dataset.user_id.unique().shape[0]
    movies = dataset.item_id.unique().shape[0]

    #Building the rating matrix
    rating_matrix = np.zeros((users,movies))

    i=0
    j=0
    z=0.0

    for tuple in dataset.itertuples():
        i = int(tuple[1]) - 1
        j = int(tuple[2]) - 1
        z = float(tuple[3])
        rating_matrix[i,j] = z



    return  rating_matrix