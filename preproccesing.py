import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split


def load_data():
    #load the train files
    df = None
    
    y_train = []

    for i in range( 5,7 ):

        tmp = pd.read_csv( 'data/train%d' % i, header=None, sep=" " )

        #tmp = pd.read_csv( 'C:/Users/Nikos/Desktop/ML_Project/ml_project/data/train%d' % i, header=None, sep=" " )
        #build labels - one hot vector
        hot_vector = [ 1 if j == i else 0 for j in range(5,7) ]
        
        for j in range( tmp.shape[0] ):
            y_train.append( hot_vector )
        #concatenate dataframes by rows    
        if i == 0:
            df = tmp
        else:
            df = pd.concat( [df, tmp] )

    train_data = df.to_numpy()
    y_train = np.array( y_train )
    
    #load test files
    df = None
    
    y_test = []

    for i in range( 5,7 ):

        tmp = pd.read_csv( 'data/test%d' % i, header=None, sep=" " )

        #tmp = pd.read_csv( 'C:/Users/Nikos/Desktop/ML_Project/ml_project/data/test%d' % i, header=None, sep=" " )
        #build labels - one hot vector
        
        hot_vector = [ 1 if j == i else 0 for j in range(5,7) ]
        
        for j in range( tmp.shape[0] ):
            y_test.append( hot_vector )
        #concatenate dataframes by rows    
        if i == 0:
            df = tmp
        else:
            df = pd.concat( [df, tmp] )

    test_data = df.to_numpy()
    y_test = np.array( y_test )
    
    return train_data, test_data, y_train, y_test


#Load the dataset.
X_train, X_test, y_train, y_test = load_data()

#split our training dataset to train and validation set.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.8, random_state=42) 




