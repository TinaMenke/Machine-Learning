# importing standard libraries
from sys import path
import os.path

# importing third party modules
# DO NOT import any other non-standard libraries: they may not be supported on Gradescope
import numpy as np
import sklearn.model_selection

import data_io
from numpy import random
from sklearn.model_selection import KFold
import datetime

def best_hyperparams():
    # Modify this dictionary to contain the best hyperparameter value(s) you found
    params = {
                "_lambda": 0,
                "_alpha": 10**(-6)
             }
    return params
    
class LinearModel:

    def __init__(self, _lambda=0, _alpha=1.0):
        self._lambda = _lambda # The regularization constant
        self._alpha = _alpha # the SGD update constant

    # fits this classifier (i.e. updates self._beta) using SGD
    # takes an array X such that X.shape = (n, P),
    #  where n is the number of samples, and P is the number of features
    # updates self._beta and returns self
    def SGD_fit(self, X, y, epochs=5): #run once

        ##### IMPLEMENT THIS FUNCTION #####

        self._beta = np.zeros(X[0].shape) # 1024
        n=X.shape[0]
        for j in np.arange(epochs):
            for i in np.arange(n): #20.000
                self._beta = self._beta-self._alpha*self.loss_grad(X[i], y[i], n)
        return self


    # predict class labels (0 for non-ships, 1 for ships)
    # takes an array X such that X.shape = (n, P),
    #  where n is the number of samples, and P is the number of features
    # also takes an optional beta parameter that can be used instead of self._beta
    # returns an array of y_i values (0 or 1) with shape (n,)
    def predict(self, X, beta=None):
        if beta is None:
            beta = self._beta
        # You don't have to edit this function, but it does call predict_prob(),
        #  so it won't work properly until you've finished that function
        p = self.predict_prob(X, beta=beta)
        return np.greater(p, 0.5*np.ones(p.shape)).astype(int)

    # return how confident the model is that each input is a ship
    # takes an array X such that X.shape = (n, P),
    #  where n is the number of samples, and P is the number of features
    # also takes an optional beta parameter that can be used instead of self._beta
    # returns an array of p_i values (between 0 and 1) with shape (n,)
    def predict_prob(self, X, beta=None):
        if beta is None:
            beta = self._beta
        ##### IMPLEMENT THIS FUNCTION #####
        ##### BE CAREFUL OF MATRIX DIMENSIONS!!! #####
        p_array = []
        p_array = 1/(1+np.exp(-np.dot(np.transpose(beta),np.transpose(X))))
        return p_array

    # computes the loss function of one data point
    # takes
    #  x, a feature vector as an array with shape (P,),
    #  y, the label for that example (0 or 1), and
    #  n, the number of samples in the entire data set
    # also takes an optional beta parameter that can be used instead of self._beta
    # returns the value for the loss, a float
    def loss(self, x, y, n, beta=None): # eigentlich x
        if beta is None:
            beta = self._beta

        p = float(self.predict_prob(np.reshape(x, (1, x.shape[0])), beta=beta))

        return -1*(y*np.log(p) + (1-y)*np.log(1-p))+(self._lambda/n)*np.sum(beta**2)

    # computes the gradient of the loss function at a given data point
    # takes x, a a feature vector as array with shape (P,),
    #  y, the example's label (0 or 1), and
    #  n, the number of samples in the entire data set
    # also takes an optional beta parameter that can be used instead of self._beta
    # returns the value for the loss, a (P,)-shaped array
    def loss_grad(self, x, y, n, beta=None):
        if beta is None:
            beta = self._beta

        ##### IMPLEMENT THIS FUNCTION #####
        p = float(self.predict_prob(np.reshape(x, (1, x.shape[0])), beta=beta))
        l_gradient = -(y-p)*x+(self._lambda/n*2)*beta
        return l_gradient
        
def cross_validation(train_x, train_y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=7)
    kf.get_n_splits(train_x)
    # arrays to save testing and training data of all five folds
    kf_X_train, kf_X_test, kf_y_train, kf_y_test = [], [], [], []

    for train_index, test_index in kf.split(train_x):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]

        kf_X_train.append(X_train)
        kf_X_test.append(X_test)
        kf_y_train.append(y_train)
        kf_y_test.append(y_test)
    return (kf_X_train, kf_X_test, kf_y_train, kf_y_test)
    
# This function will be run when this file is executed.
# It will not be executed on Gradescope
# Use this function to run your experiements
def main():

    # returns the mean L0 norm between the labels and predictions given
    # in other words, the accuracy of your model on the test data
    # labels is the ground truth labels
    # predictions is your predictions on the data
    # outputs a float between 0 and 1 (technically inclusive, but it probably won't be 0 or 1)
    def accuracy(labels, predictions):
        return ((labels == predictions).astype(int)).mean()

    train_x, train_y = data_io.read_image_data()
    # two categories instead of 4: category 2 was ships
    train_y = np.equal(train_y, 2*np.ones(train_y.shape)).astype(int)

    ##### YOUR CODE HERE ######
    lam = [0,100]
    a = [10**(-6),10**(-4),10**(-2),1,10]
    min_error = 1000000
    results = []
    time_array = []
    train_x_folds, test_x_folds, train_y_folds, test_y_folds = cross_validation(train_x, train_y)
    for i in lam:
        for j in a:
        	time_start=datetime.datetime.now()
        	error_array = []
        	for k in np.arange(5):
        		classifier = LinearModel(_lambda=i, _alpha=j)
        		classifier.SGD_fit(train_x_folds[k], train_y_folds[k])
        		prediction = classifier.predict(test_x_folds[k])
        		error = 1-accuracy(prediction, test_y_folds[k])
        		error_array = np.append(error_array, error)
        	ac_error = np.mean(error_array)
        	results = np.append(results, ac_error)
        	time_end = datetime.datetime.now()
        	time_delta = (time_end-time_start).total_seconds()
        	time_array = np.append(time_array, time_delta)
        	if(ac_error<min_error):
        		min_error = ac_error
        		best_model = classifier
        		best_l = i
        		best_a = j
    print(time_array)
    
    #Table with out of sample errors
    print("Table with out of sample errors:")
    out_of_sample_error = results
    print("lambda", '\t', "alpha", '\t', "error")
    print("-----", '\t', "-----", '\t', "-----")
    k=0
    for i in range(2):
    	for j in range(5):
    			print(lam[i], '\t', a[j], '\t', results[k])
    			k = k+1
    
    print("optimal parameter: lambda = " + str(best_l) + " and alpha = " + str(best_a))
   
    
    # create the classifier
    #classifier = LinearModel(_lambda=0, _alpha=1.0)

    # You'll probably want to change this line for cross-validation
    #classifier.SGD_fit(train_x, train_y)

    # This is a dummy test data set.
    # They're all just random noise, not ships or horses or frogs or trucks,
    # so your classifier should and will perform poorly on these.
    # These lines of code are for demonstration purposes only.
    #test_x = random.random(train_x.shape)
    #test_y = random.randint(0, 2, size=train_y.shape)
    #predictions = classifier.predict(test_x)

    # prints the accuracy of your predictions
    # should be about 0.5 for the random test data above
    #print(accuracy(predictions, test_y))
    
    # predict on full test data
    test_x, test_y = data_io.read_test_data()
    test_y = np.equal(test_y, 2*np.ones(test_y.shape)).astype(int)
    predictions = best_model.predict(test_x)
    print("Test error: " + str(1-accuracy(predictions, test_y)))

    # While it's good practice to import at the beginning of a file,
    # since Gradescope won't run this function,
    # you can import anything you want here.
    # matplotlib's pyplot is a good tool for making plots.
    # You can install it here: http://lmgtfy.com/?q=matplotlib+download+python3+anaconda
    # You can read about it here: http://lmgtfy.com/?q=matplotlib+pyplot+tutorial
    from matplotlib import pyplot
    combinations = ["0/10^(-6)", "0/10^(-4)", "0/0.01", "0/1", "0/10", "100/10^(-6)", "100/10^(-4)", "100/0.01", "100/1", "100/10"]
    pyplot.bar(combinations, results)
    pyplot.ylabel("out of sample error")
    pyplot.xlabel("lambda/alpha value")
    pyplot.title("Out of sample error for different lambda and alpha")
    pyplot.show()

# This runs 'main' upon loading the script
if __name__ == '__main__':
    main()

