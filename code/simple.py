## Simple Regression Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt


# Read data matrix X and labels t from text file.
def read_data(file_name):

#  YOUR CODE here:
  data_file = np.loadtxt(file_name)
  X = data_file[:, 0] # square ft
  t = data_file[:, 1]/100 # price

  return X, t


# Implement simple linear regression to compute w = [w0, w1].
def train(X, t):
#  YOUR CODE here:
 
  N = len(t) # Length of t stored in N
  coefficent  = np.array([[N, np.sum(X)], [np.sum(X), np.sum(X**2)]] ) # The coefficent of the equation
  dependent = np.array([np.sum(t), np.sum(t * X)]) # the dependent variables in the equation
  w = np.linalg.solve(coefficent, dependent) # The final calculation stored in w 

  return w

 
# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):
#  YOUR CODE here:
  N = len(X) # Length of x, stored in N
  X = np.column_stack((np.ones(N), X))
  top = np.sum((np.dot(X, w) - t)**2) # calculates the numerator portion of the equation
  rmse = np.sqrt(top/N) # calculates the whole function by taking the square root and dividing by N
  
  return rmse


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
#  YOUR CODE here:
  N = len(X) # Length of X square ft stored in N
  X = np.column_stack((np.ones(N), X))
  first = (1)/(2*N) # The first part of the equation 1/2N
  second = np.sum((np.dot(X, w) - t)**2) # second part of equation which consists of sum and squaring it
  cost = first * second # The cost function all together to get the actual value
  return cost


##======================= Main program =======================##
parser = argparse.ArgumentParser('Simple Regression Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/simple',
                    help='Directory for the simple houses dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")

# Train model on training examples.
w = train(Xtrain, ttrain)

# Print model parameters.
print('Params: ', w)

# Print cost and RMSE on training data.
print('Training RMSE: %0.2f.' % compute_rmse(Xtrain, ttrain, w))
print('Training cost: %0.2f.' % compute_cost(Xtrain, ttrain, w))

# Print cost and RMSE on test data.
print('Test RMSE: %0.2f.' % compute_rmse(Xtest, ttest, w))
print('Test cost: %0.2f.' % compute_cost(Xtest, ttest, w))

#  YOUR CODE here: plot the training and test examples with different symbols,
#                  plot the linear approximation on the same graph.


plt.title("Xtrain Vs ttrain")
plt.scatter(Xtest, ttest, marker = '^', color = 'limegreen') # the test data
plt.scatter(Xtrain, ttrain) # the train data
plt.xlabel("Xtrain")
plt.ylabel("ttrain")

# Plotting a line
x = np.array([500,4700])
x_matrix = np.column_stack((np.array([1, 1]), np.array([500,4700])))
plt.plot(x, x_matrix.dot(w), color = 'red')

# Save data
plt.savefig('train-test-line.png')


