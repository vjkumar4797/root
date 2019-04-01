import numpy as np
import scipy.stats as stats
import csv

#function to obfuscate cluster centres

def obfuscate(centres , r_matrix):
    #temporary arrays to store kendall tau distances and sigma
    t  = np.zeros(5)
    s = np.zeros(5)

    #temporary array to store noisy centres
    noisy_centres = np.zeros(centres.shape)
    n_centre = np.zeros(centres[0].shape)


    for i in range(0 , 5):
        #Initial values for mean and standard deviation to be given to the gaussian function to generate noise
        mu , sigma = 0 , 0.1
        tau = 1

        while(tau>0.5):
            prev_cn = n_centre
            prev_t = tau

            #Generate Noise
            noise = np.random.normal(mu, sigma, centres[i].shape)

            #Add noise to the ith cluster centre
            n_centre = centres[i] + noise

            #print(n_centre)

            #Kendall tau metric to determine the ranking loss between the noisy centres and actual centres
            tau , p_value = stats.kendalltau(centres[i] , n_centre)

            #Store the current standard deviation value in a temporary variable prev_sigma
            prev_s = sigma
            sigma += 0.01

        #print(prev_t , prev_s)
        noisy_centres[i] = prev_cn
        t[i] = prev_t
        s[i] = prev_s

    #print(centres)
    #print(noisy_centres)
    users , movies = r_matrix.shape
    noisy_matrix = np.zeros(r_matrix[: , :-1].shape)
    for i in range(0 , users):
        cluster = int( r_matrix[i , movies-1])
        usr_vector = r_matrix[i , :-1]
        sigma = s[cluster]
        noise = np.random.normal( 0 , sigma , usr_vector.shape)
        noisy_matrix[i] = usr_vector + noise
        tau , p_value = stats.kendalltau(noisy_matrix[i] , usr_vector)


    count = 0
    csv.register_dialect('myDialect', delimiter='\t', lineterminator='\n')

    with open('noisy_dataset2`.csv', 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        for i in range(0 , users):
            for j in range( 0 , movies - 1 ):
                data = []
                if(r_matrix[i][j] == 0.0):
                    noisy_matrix[i][j] = 0.0
                elif(noisy_matrix[i][j]>5):
                    noisy_matrix[i][j] = 4.5
                    data.append(i+1)
                    #data.append('\t')
                    data.append(j+1)
                    #data.append('\t')
                    data.append(noisy_matrix[i][j])
                    writer.writerow(data)
                    count = count+1
                else:
                    data.append(i+1)
                    #data.append('\t')
                    data.append(j+1)
                    #data.append('\t')
                    data.append(noisy_matrix[i][j])
                    writer.writerow(data)
                    count = count+1






    f.close()



    tau  , p_value = stats.kendalltau(noisy_matrix , r_matrix[: , :-1])