import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data():
    #load the train files
    df = None
    
    y_train = []

    for i in range( 5,7 ):
        
        tmp = pd.read_csv( 'data/train%d.txt' % i, header=None, sep=" " )
                
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
        
        tmp = pd.read_csv( 'data/test%d.txt' % i, header=None, sep=" " )
        
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

#Preprocessing

#Reshaping our images from 28*28 to 784-element vectors.

def create_feature_vectors(num_of_features, data):
    
    vector = data.reshape([-1, num_of_features])
    return vector



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

#We won't use L2 regularization on this version
def calculateCost(X, y, theta):
    
    h = sigmoid(X.dot(theta))
    cur_J = (y.T.dot(np.log(h + epsilon)) + (1-y).T.dot(np.log(1 - h + epsilon)))
    
    #calculate gradient
    gradient = X.T.dot(y-h)#.reshape(X.shape[1],1)
    #print(gradient.shape)
    
    return cur_J, gradient

def LogisticRegression(X, y, X_val, y_val, epoch = 2000, alpha = 0.01):
    
    #theta = np.random.normal(size=(X.shape[1],1))
    theta = np.zeros(X.shape[1]).reshape( (-1,1) )

    #print("theta.shape is: ", theta.shape)
    m, n = X.shape
    
    J_train = []
    J_test = []
    
    for i in range(epoch):
        train_error, train_gradient = calculateCost(X, y, theta)
        #print("I've passed from train.")
        test_error, _ = calculateCost(X_val, y_val, theta)
        #print("I've passed from test.")

        
        #update theta parameters, using gradient ascent.
        theta += alpha * train_gradient
        
        J_train.append(train_error[0])
        J_test.append(test_error[0])
        
    return J_train, J_test, theta





