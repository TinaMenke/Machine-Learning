# importing standard libraries
from sys import path
import os.path

# importing third party modules
# DO NOT import any other non-standard libraries: they may not be supported on Gradescope
import numpy as np
import sklearn.tree
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from numpy import random

# importing local files
path.append(os.path.join(os.path.realpath(__file__), os.pardir))
import data_io
import datetime

def best_hyperparams():
    # Modify this dictionary to contain the best hyperparameter value(s) you found
    params = {
                "max_depth": 9
             }
    return params["max_depth"]

# This function will be executed by Gradescope
# You should probably also call it from main()
# Here is where you should create and train your decision tree
# train_x and train_y are the training data (numpy arrays)
# hyperparams is a dictionary of hyperparameters (in this case just 'max_depth')
#  see best_hyperparams() above for an example
# This function should return a trained decision tree
def create_model(train_x, train_y, hyperparams):

    # 'classifier' should be the decision tree model you train
    # See here: http://lmgtfy.com/?q=sklearn+decision+tree+classifier
    classifier = DecisionTreeClassifier(max_depth = hyperparams, random_state=7)
    classifier.fit(train_x,train_y)

    ##### YOUR CODE HERE #####

    return classifier
    
def cross_validation(train_x, train_y, k=5):
	kf = KFold(n_splits=k, shuffle = True, random_state = 7)
	kf.get_n_splits(train_x) 
	kf_X_train, kf_X_test, kf_y_train, kf_y_test = [], [], [], []
	
	for train_index, test_index in kf.split(train_x):
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

    ##### YOUR CODE HERE ######

    # You'll probably want to change this line for cross-validation
    # This is just an example of a call to create_model, defined above
    
    train_x_folds, test_x_folds, train_y_folds, test_y_folds = cross_validation(train_x, train_y)
    
    
    time_array = []
    results = []
    depth_of_DT=[3,6,9,12,14]
    min_error = 1000000
    
    for i in depth_of_DT:
    	time_start = datetime.datetime.now()
    	error_array = []
    	for k in range(5):
    		classifier = create_model(train_x_folds[k], train_y_folds[k], i)
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
    		best_parameter = i
    print(time_array)
    print("optimal parameter: " + str(best_parameter))
    
    	
    #Table with out of sample errors
    print("Table with out of sample errors for depth = 3,6,9,12,14:")
    out_of_sample_error = results
    print("depth", '\t', "error")
    print("-----", '\t', "-----")
    for i in range(5):
    	print(depth_of_DT[i], '\t', out_of_sample_error[i])

    # This is a dummy test data set.
    # They're all just random noise, not ships or horses or frogs or trucks,
    # so your classifier should and will perform poorly on these.
    # These lines of code are for demonstration purposes only.
    #test_x = random.random(train_x.shape)
    #test_y = random.randint(0, 4, size=train_y.shape)
    #predictions = classifier.predict(test_x)

    # prints the accuracy of your predictions
    # should be about 0.25 for the random test data above
    #print(accuracy(predictions, test_y))
    
    #predict on full test data
    test_x, test_y = data_io.read_test_data() 
    predictions = best_model.predict(test_x)
    print("Test error: " + str(1-accuracy(predictions, test_y)))

    # While it's good practice to import at the beginning of a file,
    # since Gradescope won't run this function,
    # you can import anything you want here.
    # matplotlib's pyplot is a good tool for making plots.
    # You can install it here: http://lmgtfy.com/?q=matplotlib+download+python3+anaconda
    # You can read about it here: http://lmgtfy.com/?q=matplotlib+pyplot+tutorial
    from matplotlib import pyplot
    depth = [3,6,9,12,14]
    pyplot.bar(depth, results)
    pyplot.ylabel("out of sample error")
    pyplot.xlabel("maximum depth")
    pyplot.title("Out of sample error for maximum depth = 3,6,9,12,14")
    pyplot.show()
    

# This runs 'main' upon loading the script
if __name__ == "__main__":
    main()
