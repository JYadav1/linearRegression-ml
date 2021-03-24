# # Polynomial Curve Fitting
import math
import argparse
import sys
from numpy import linalg as la
import numpy as np
from matplotlib import pyplot as plt

# # Part A # #

# Read data matrix X and labels t from text file.
def read_data(file_name):
    data_file = np.loadtxt(file_name)
    X = data_file[:, 0] # square ft
    t = data_file[:, 1] # price
    return X, t

##======================= Main program =======================##
parser = argparse.ArgumentParser('Polynomial Curve Fitting')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/polyfit',
                    help='Directory for the houses dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training data.
X_dataset, t_dataset = read_data(FLAGS.input_data_dir + "/dataset.txt")
X_train, t_train = read_data(FLAGS.input_data_dir + "/train.txt")
X_test, t_test = read_data(FLAGS.input_data_dir + "/test.txt")
X_devel, t_devel = read_data(FLAGS.input_data_dir + "/devel.txt")


# # Plotting  for 3a # #

# plot for dataset, Part A
plt.figure(0)
plt.title('X_dataset vs t_dataset')
plt.scatter(X_dataset, t_dataset, color = 'blue') # the dataset data
plt.xlabel("X_dataset")
plt.ylabel("t_dataset")
plt.savefig('X-t-dataset.png')
# # Part B # #

# # Plotting for 3b # #

# plot for train data
plt.figure(1)
plt.title('X_train vs t_train')
plt.scatter(X_train, t_train, color = 'blue') # the train data
plt.xlabel("X_train")
plt.ylabel("t_train")
plt.savefig('X-t-train.png')

# plot for test data
plt.figure(2)
plt.title('X_test vs t_test')
plt.scatter(X_test, t_test, color = 'blue') # the test data
plt.xlabel("Xtest")
plt.ylabel("ttest")
plt.savefig('X-t-test.png')

# plot for devel data
plt.figure(3)
plt.title('X_devel vs t_devel')
plt.scatter(X_devel, t_devel, color = 'blue') # the devel data
plt.xlabel("X_devel")
plt.ylabel("t_devel")
plt.savefig('X-t-devel.png')

# # Part c is done in the report # #


# # part d # #
def compute_func_train(X, t):
    trans_x = X.T
    inverse = np.linalg.inv(trans_x @ X)
    w = (inverse @ (trans_x @ t))
    return w

def compute_rmse_func(X, t, w):
    N = len(X)
    top = np.sum((np.dot(X, w) - t)**2) # calculates the numerator portion of the equation
    rmse = np.sqrt(top/N) # calculates the whole function by taking the square root and dividing by N
    return rmse

def compute_regular_rmse(X, t, w, ln_lambda):
    lam = math.exp(ln_lambda)
    N = len(X)
    top = np.sum(((X @ w + ((lam/2) * la.norm(w)**2)) - t)**2) # calculates the numerator portion of the equation
    rmse = np.sqrt(top/N) # calculates the whole function by taking the square root and dividing by N
    return rmse


def create_x_matrix(X, M):
    X = X[:, np.newaxis]
    X_column = np.power(X, 0)
    for i in range(1, M+1):
        X_column = np.column_stack((X_column, np.power(X, i)))
    return X_column

# # polynomial curve fitting  # #
# # Part 1 # #

rmse_train = []
rmse_test = []
xMatrices = [None] * 10

for j in range(0, 10):
    xMatrices[j] = create_x_matrix(X_train, j)

for i in range(0, 10):
    w = compute_func_train(xMatrices[i], t_train)
    rmse_train.append(compute_rmse_func(xMatrices[i], t_train, w))

for k in range(0, 10):
    w = compute_func_train(xMatrices[k], t_train)
    rmse_test.append(compute_rmse_func(xMatrices[k], t_test, w))

# # plot Part 1 D # #
# # Train Data # #
plt.figure(4)
ten = np.arange(10)
plt.title('graph 1d')
plt.plot(ten, rmse_train, color = 'black', linestyle = 'solid', label = 'Training') # the train data
plt.xlabel("M (degree)")
plt.ylabel("RMSE")

# # Test Data # #
plt.plot(ten, rmse_test, color = 'yellow', linestyle = 'solid', label = 'Testing') # the test data
leg = plt.legend()
plt.savefig('3d-part1.png')

# # polynomial curve fitting  # #
# # Part 2 # #

rmse_training = []
rmse_devel = []

for i in range(-50, 5, 5):
    create_x_matrix(X_test, 9)
    rmse_training.append(compute_rmse_func(create_x_matrix(X_test, 9), t_test, i))









        


