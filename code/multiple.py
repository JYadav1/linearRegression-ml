## Multiple Regression Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt


# Read data matrix X and labels t from text file.
def read_data(file_name):
#  YOUR CODE here:
  data_file = np.loadtxt(file_name)
  X = data_file[:, 0:3] # square ft
  t = (data_file[:, 3])/100 # price

  return X, t


# Implement normal equations to compute w = [w0, w1, ..., w_M].
def train(X, t):
#  YOUR CODE here:
  trans_x = X.T
  inverse = np.linalg.inv(trans_x @ X)
  w = (inverse @ (trans_x @ t))
  return w


# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):
#  YOUR CODE here:
  N = len(X)
  top = np.sum((np.dot(X, w) - t)**2) # calculates the numerator portion of the equation
  rmse = np.sqrt(top/N) # calculates the whole function by taking the square root and dividing by N
  return rmse


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
#  YOUR CODE here:
  N = len(X)
  first = (1)/(2*N) # The first part of the equation 1/2N
  second = np.sum((np.dot(X, w) - t)**2) # second part of equation which consists of sum and squaring it
  cost = first * second # The cost function all together to get the actual value
  return cost



##======================= Main program =======================##
parser = argparse.ArgumentParser('Multiple Regression Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/multiple',
                    help='Directory for the multiple houses dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")

#  YOUR CODE here: add the bias feature to each training and test example,
#                  create new design matrices X1train and X1test.
ones = np.ones(Xtrain.shape[0]) 
ones_test = np.ones(Xtest.shape[0])
X1train = np.column_stack((ones, Xtrain))
X1test = np.column_stack((ones_test, Xtest))



# Train model on training examples.
w = train(X1train, ttrain)

# Print model parameters.
print('Params: ', w, '\n')

# Print cost and RMSE on training data.
print('Training RMSE: %0.2f.' % compute_rmse(X1train, ttrain, w))
print('Training cost: %0.2f.' % compute_cost(X1train, ttrain, w))

# Print cost and RMSE on test data.
print('Test RMSE: %0.2f.' % compute_rmse(X1test, ttest, w))
print('Test cost: %0.2f.' % compute_cost(X1test, ttest, w))
