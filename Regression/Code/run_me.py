# Import python modules
import numpy as np
import kaggle

# Read in train and test data
def read_data_power_plant():
	print('Reading power plant dataset ...')
	train_x = np.loadtxt('../../Data/PowerOutput/data_train.txt')
	train_y = np.loadtxt('../../Data/PowerOutput/labels_train.txt')
	test_x = np.loadtxt('../../Data/PowerOutput/data_test.txt')

	return (train_x, train_y, test_x)

def read_data_localization_indoors():
	print('Reading indoor localization dataset ...')
	train_x = np.loadtxt('../../Data/IndoorLocalization/data_train.txt')
	train_y = np.loadtxt('../../Data/IndoorLocalization/labels_train.txt')
	test_x = np.loadtxt('../../Data/IndoorLocalization/data_test.txt')

	return (train_x, train_y, test_x)

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

############################################################################

train_x, train_y, test_x = read_data_power_plant()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

train_x, train_y, test_x = read_data_localization_indoors()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Create dummy test output values for IndoorLocalization
predicted_y = np.ones(test_x.shape[0]) * -1
# Output file location
file_name = '../Predictions/IndoorLocalization/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

# Create dummy test output values for PowerOutput
predicted_y = np.ones(test_x.shape[0]) * -1
# Output file location
file_name = '../Predictions/PowerOutput/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

############################################################################

#imports
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from datascience import Table
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

############################################################################

# Cross validation Implementation
# Implement a cross validation function

def cross_validation(train_x, train_y, k=5):
	kf = KFold(n_splits=k, shuffle=True, random_state=0)
	kf.get_n_splits(train_x)
	# arrays to save testing and training data of all five folds
	kf_X_train, kf_X_test, kf_y_train, kf_y_test = [], [], [], []

	# print(kf)

	for train_index, test_index in kf.split(train_x):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = train_x[train_index], train_x[test_index]
		y_train, y_test = train_y[train_index], train_y[test_index]

		kf_X_train.append(X_train)
		kf_X_test.append(X_test)
		kf_y_train.append(y_train)
		kf_y_test.append(y_test)

	return (kf_X_train, kf_X_test, kf_y_train, kf_y_test)
	
############################################################################

# Data Preprocessing

# get power plant data
pp_train_x, pp_train_y, pp_test_x = read_data_power_plant()

# devide power plant data in 5 folds
pp_train_x_folds, pp_test_x_folds, pp_train_y_folds, pp_test_y_folds = cross_validation(pp_train_x, pp_train_y)

# get indoor localization data
il_train_x, il_train_y, il_test_x = read_data_localization_indoors()

# divide power plant data in 5 folds
il_train_x_folds, il_test_x_folds, il_train_y_folds, il_test_y_folds = cross_validation(il_train_x, il_train_y)
############################################################################

# DT for Power Plant Dataset

# Function to make predictions with Decision tree model
def decision_tree_prediction(x_train,y_train,x_test,a):
    regressor = DecisionTreeRegressor(criterion="mae",max_depth=a, random_state=0)
    regressor.fit(x_train, y_train)
    return regressor.predict(x_test)
    
# Function to calculate average error of five folds
def dt_calculate_fold_error(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds, depth):
    MAE_error = []
    for i in range(0,5):
        prediction = decision_tree_prediction(pp_train_x_folds[i],pp_train_y_folds[i],pp_test_x_folds[i],depth)
        MAE_error.append(compute_error(prediction, pp_test_y_folds[i]))
    return np.mean(MAE_error)

# Function to calculate error for models with depth = 3,6,9,12,15
def dt_calculate_error_for_depth(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds):
    error_array = []
    time_array = []
    for i in (3,6,9,12,15):
        time_start = datetime.datetime.now()
        error = dt_calculate_fold_error(pp_train_x_folds,pp_train_y_folds, pp_test_x_folds, pp_test_y_folds, i)
        time_end = datetime.datetime.now()
        time_delta = (time_end-time_start).total_seconds()*1000 # time delta in milliseconds
        error_array = np.append(error_array, error)
        time_array = np.append(time_array, time_delta)
    return error_array, time_array

# Calulate out of sample error for all depths
error, t = dt_calculate_error_for_depth(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds)

# Plot graph reporting the time in ms for cross validation
depth = [3,6,9,12,15]
fig, ax = plt.subplots()
ax.plot(depth, t)
ax.set(xlabel='depth of decision tree', ylabel='time (ms)',
       title='Graph reporting the time in ms for cross validation')
ax.grid()
fig.savefig("../Figures/PowerOutput/DT.png")
plt.show()

# Create table with out of sample errors
print("")
print("[DT for Power plant data]")
print("Table with out of sample errors for depth = 3, 6, 9, 12, 15:")
t3 = Table()
t3['depth of DT'] = [3,6,9,12,15]
t3['average error'] = error
print(t3)

# Choose the model with lowest estimated out of sample error
min_error = t3['average error'].min()
minimum = t3.where('average error', min_error)
max_depth = int(minimum['depth of DT'])
print("For the model with the lowest estimated out of sample error depth equals: " + str(max_depth))

# train best model with the full training set
predicted_y = decision_tree_prediction(pp_train_x,pp_train_y,pp_test_x,max_depth)

# Kaggleize predictions
#file_name = '../Predictions/PowerOutput/DT.csv'
#kaggle.kaggleize(predicted_y, file_name)

############################################################################

# DT for Indoor Localization Dataset

# Function to calculate error for models with depth = 20,25,30,35,40
def dt_il_calculate_error_for_depth(il_train_x_folds, il_train_y_folds, il_test_x_folds, il_test_y_folds):
    error_array = []
    time_array = []
    for i in (20, 25, 30, 35, 40):
        time_start = datetime.datetime.now()
        error = dt_calculate_fold_error(il_train_x_folds,il_train_y_folds, il_test_x_folds, il_test_y_folds, i)
        time_end = datetime.datetime.now()
        time_delta = (time_end-time_start).total_seconds()*1000 # time delta in milliseconds
        error_array = np.append(error_array, error)
        time_array = np.append(time_array, time_delta)
    return error_array, time_array
    
# Calulate out of sample error for all depths
error, t = dt_il_calculate_error_for_depth(il_train_x_folds, il_train_y_folds, il_test_x_folds, il_test_y_folds)

# Plot graph reporting the time in ms for cross validation
depth = [20,25,30,35,40]
fig, ax = plt.subplots()
ax.plot(depth, t)
ax.set(xlabel='depth of decision tree', ylabel='time (ms)',
       title='Graph reporting the time in ms for cross validation')
ax.grid()
fig.savefig("../Figures/IndoorLocalization/DT.png")
plt.show()

# Create table with out of sample errors
print("")
print("[DT for Indoor Localization data]")
print("Table with out of sample errors for depth = 20, 25, 30, 35, 40:")
t4 = Table()
t4['depth of DT'] = [20, 25, 30, 35, 40]
t4['average error'] = error
print(t4)

# Choose the model with lowest estimated out of sample error
min_error = t4['average error'].min()
minimum = t4.where('average error', min_error)
max_depth = int(minimum['depth of DT'])
print("For the model with the lowest estimated out of sample error depth equals: " + str(max_depth))

# train best model with the full training set
predicted_y = decision_tree_prediction(il_train_x,il_train_y,il_test_x,max_depth)

# Kaggleize predictions
#file_name = '../Predictions/IndoorLocalization/DT.csv'
#kaggle.kaggleize(predicted_y, file_name)
############################################################################

# kNN for power plant data

# Function to make predictions with KNeighborsRegressor Model
def knn_prediction(train_x,train_y,test_x,k):
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(train_x, train_y)
    return neigh.predict(test_x)
    
# Function to calculate average error of five folds
def calculate_fold_error(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds,k):
    MAE_error = []
    for i in range(0,5):
        prediction = knn_prediction(pp_train_x_folds[i],pp_train_y_folds[i],pp_test_x_folds[i],k)
        MAE_error.append(compute_error(prediction, pp_test_y_folds[i]))
    return np.mean(MAE_error)

# Function to calculate error for models with k = 3, 5, 10, 20, 25
def calculate_k_error(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds):
    error_array = []
    for i in (3, 5, 10, 20, 25):
        error = calculate_fold_error(pp_train_x_folds,pp_train_y_folds, pp_test_x_folds, pp_test_y_folds,i)
        error_array = np.append(error_array, error)
    return error_array
    
# Calulate out of sample error for all k
error = calculate_k_error(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds)

# Create table with out of sample errors
print("")
print("[kNN for Power plant data]")
print("Table with out of sample errors for k = 3, 5, 10, 20, 25:")
t1 = Table()
t1['k value'] = [3, 5, 10, 20, 25]
t1['average error'] = error
print(t1)

# Choose the model with lowest estimated out of sample error
min_error = t1['average error'].min()
minimum = t1.where('average error', min_error)
min_k = int(minimum['k value'])
print("For the model with the lowest estimated out of sample error k equals: " + str(min_k))

# train best model with the full training set
predicted_y = prediction_best_model = knn_prediction(pp_train_x,pp_train_y,pp_test_x,min_k)

# Kaggleize kNN predictions
#file_name = '../Predictions/PowerOutput/kNN.csv'
#kaggle.kaggleize(predicted_y, file_name)
############################################################################

# kNN for Indoor Localization dataset

# Calulate out of sample error for all k
error = calculate_k_error(il_train_x_folds, il_train_y_folds, il_test_x_folds, il_test_y_folds)

# Create table with out of sample errors
print("")
print("[kNN for Indoor localization data]")
print("Table with out of sample errors for k = 3, 5, 10, 20, 25:")
t2 = Table()
t2['k value'] = [3,5,10,20,25]
t2['average error'] = error
print(t2)

# Choose the model with lowest estimated out of sample error
min_error = t2['average error'].min()
minimum = t2.where('average error', min_error)
min_k = int(minimum['k value'])
print("For the model with the lowest estimated out of sample error k equals: " + str(min_k))

# train best model with the full training set
predicted_y = prediction_best_model = knn_prediction(il_train_x,il_train_y,il_test_x,min_k)

# Kaggleize kNN predictions
#file_name = '../Predictions/IndoorLocalization/kNN.csv'
#kaggle.kaggleize(predicted_y, file_name)
############################################################################

# Ridge & Lasso for Power Plant Dataset

# Function to make predictions with Ridge Linear model
def ridge_linear_prediction(X_train, y_train, x_test, a): 
    clf = Ridge(alpha=a, random_state=0) # alpha=a, random_state=0, normalize= True
    clf.fit(X_train, y_train) 
    return clf.predict(x_test)
    
# calculate average error of folds
def ridge_calculate_fold_error(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds,a):
    MAE_error = []
    for i in range(0,5):
        prediction = ridge_linear_prediction(pp_train_x_folds[i],pp_train_y_folds[i],pp_test_x_folds[i],a)
        MAE_error.append(compute_error(prediction, pp_test_y_folds[i]))
    return np.mean(MAE_error)
    
def ridge_calculate_error_for_alpha(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds):
    error_array = []
    for a in (0.000001, 0.0001, 0.01, 1, 10):
        error = ridge_calculate_fold_error(pp_train_x_folds,pp_train_y_folds, pp_test_x_folds,pp_test_y_folds,a)
        error_array = np.append(error_array, format(error, '.13'))
    return error_array
    
# Function to make predictions with Lasso Linear model
def lasso_linear_prediction(X_train, y_train, x_test, a): 
    clf = linear_model.Lasso(alpha=a, random_state=0) # alpha=a, random_state=0, normalize=True
    clf.fit(X_train, y_train) 
    return clf.predict(x_test)
    
# calculate average error of folds
def lasso_calculate_fold_error(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds,a):
    MAE_error = []
    for i in range(0,5):
        prediction = lasso_linear_prediction(pp_train_x_folds[i],pp_train_y_folds[i],pp_test_x_folds[i],a)
        MAE_error.append(compute_error(prediction, pp_test_y_folds[i]))
    return np.mean(MAE_error)
    
def lasso_calculate_error_for_alpha(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds):
    error_array = []
    for a in (0.000001, 0.0001, 0.01, 1, 10):
        error = lasso_calculate_fold_error(pp_train_x_folds,pp_train_y_folds, pp_test_x_folds,pp_test_y_folds,a)
        error_array = np.append(error_array, format(error, '.13f'))
    return error_array
    
error_ridge = ridge_calculate_error_for_alpha(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds)
error_lasso = lasso_calculate_error_for_alpha(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds)

print("")
print("[Ridge & Lasso Linear Regression for Power Plant data]")
print("Table with out of sample errors for α = 0.000001, 0.0001, 0.01, 1, 10:")
t5 = Table()
t5['alpha'] = [10**(-6),10**(-4),10**(-2),1,10]
t5['Ridge error'] = error_ridge
t5['Lasso error'] = error_lasso
print(t5)
min_error = min(min(error_ridge),min(error_lasso))
if min(error_ridge)<min(error_lasso):
    minimum = t5.where('Ridge error', min_error)
    alpha = float(minimum['alpha'])
    predicted_y = ridge_linear_prediction(pp_train_x,pp_train_y,pp_test_x,alpha)
else:
    minimum = t5.where('Lasso error', min_error)
    alpha = float(minimum['alpha'])
    predicted_y = lasso_linear_prediction(pp_train_x,pp_train_y,pp_test_x,alpha)
print("For the model with the lowest estimated out of sample error alpha equals: " + str(alpha))

# Kaggleize predictions
#file_name = '../Predictions/PowerOutput/Ridge&Lasso.csv'
#kaggle.kaggleize(predicted_y, file_name)

############################################################################

# function to calculate sample error for ridge model
def ridge_calculate_error_for_alpha_il(il_train_x_folds, il_train_y_folds, il_test_x_folds, il_test_y_folds):
    error_array = []
    time_array = []
    for a in (0.0001, 0.01, 1, 10):
        error = ridge_calculate_fold_error(il_train_x_folds,il_train_y_folds, il_test_x_folds,il_test_y_folds,a)
        error_array = np.append(error_array, format(error, '.13'))
    return error_array
    
# function to calculate sample error for lasso model
def lasso_calculate_error_for_alpha_il(il_train_x_folds, il_train_y_folds, il_test_x_folds, il_test_y_folds):
    error_array = []
    for a in (0.0001, 0.01, 1, 10):
        error = lasso_calculate_fold_error(il_train_x_folds,il_train_y_folds, il_test_x_folds,il_test_y_folds,a)
        error_array = np.append(error_array, format(error, '.13'))
    return error_array
    
error_ridge = ridge_calculate_error_for_alpha_il(il_train_x_folds, il_train_y_folds, il_test_x_folds, il_test_y_folds)
error_lasso = lasso_calculate_error_for_alpha_il(il_train_x_folds, il_train_y_folds, il_test_x_folds, il_test_y_folds)

print("")
print("[Ridge & Lasso Linear Regression for Indoor Localization data]")
print("Table with out of sample errors for α = 0.0001, 0.01, 1, 10:")
t6 = Table()
t6['alpha'] = [10**(-4),10**(-2),1,10]
t6['Ridge error'] = error_ridge
t6['Lasso error'] = error_lasso
print(t6)
min_error = min(min(error_ridge),min(error_lasso))
if min(error_ridge)<min(error_lasso):
    minimum = t6.where('Ridge error', min_error)
    alpha = float(minimum['alpha'])
    predicted_y = ridge_linear_prediction(il_train_x,il_train_y,il_test_x,alpha)
else:
    minimum = t6.where('Lasso error', min_error)
    alpha = float(minimum['alpha'])
    predicted_y = lasso_linear_prediction(il_train_x,il_train_y,il_test_x,alpha)
print("For the model with the lowest estimated out of sample error alpha equals: " + str(alpha))

# Kaggleize predictions
#file_name = '../Predictions/IndoorLocalization/Ridge&Lasso.csv'
#kaggle.kaggleize(predicted_y, file_name)

############################################################################

# Best model for Power Plant Data

# Function to make predictions with Decision tree model
def best_decision_tree_prediction(x_train,y_train,x_test,a):
    regressor = DecisionTreeRegressor(criterion="mae",max_depth=a, random_state=0)
    regressor.fit(x_train, y_train)
    return regressor.predict(x_test)
    
# Function to calculate average error of five folds
def best_calculate_fold_error(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds, depth):
    MAE_error = []
    for i in range(0,5):
        prediction = best_decision_tree_prediction(pp_train_x_folds[i],pp_train_y_folds[i],pp_test_x_folds[i],depth)
        MAE_error.append(compute_error(prediction, pp_test_y_folds[i]))
    return np.mean(MAE_error)
    
# Function to calculate error for models with depth = 8,9,10,11,12
def best_error_for_depth(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds):
    error_array = []
    for i in (8,9,10,11,12):
        error = best_calculate_fold_error(pp_train_x_folds,pp_train_y_folds, pp_test_x_folds, pp_test_y_folds, i)
        error_array = np.append(error_array, error)
    return error_array
    
error = best_error_for_depth(pp_train_x_folds, pp_train_y_folds, pp_test_x_folds, pp_test_y_folds)

# Create table with out of sample errors
print("")
print("[Best model for Power Plant data]")
print("Table with out of sample errors for depth = 8,9,10,11,12:")
t7 = Table()
t7['depth of DT'] = [8,9,10,11,12]
t7['average error'] = error
print(t7)

# Choose the model with lowest estimated out of sample error
min_error = t7['average error'].min()
minimum = t7.where('average error', min_error)
max_depth = int(minimum['depth of DT'])
print("For the model with the lowest estimated out of sample error depth equals: " + str(max_depth))

predicted_y = decision_tree_prediction(pp_train_x,pp_train_y,pp_test_x,max_depth)

# Kaggleize predictions
file_name = '../Predictions/PowerOutput/best.csv'
kaggle.kaggleize(predicted_y, file_name)

############################################################################

# Best model for Indoor Localization Data

# Function to calculate average error of five folds
def calculate_fold_error(il_train_x_folds, il_train_y_folds, il_test_x_folds, il_test_y_folds,k):
    MAE_error = []
    for i in range(0,5):
        prediction = knn_prediction(il_train_x_folds[i],il_train_y_folds[i],il_test_x_folds[i],k)
        MAE_error.append(compute_error(prediction, il_test_y_folds[i]))
    return np.mean(MAE_error)
    
# Function to calculate error for models with k = 1,2,3,4,5
def k_error(il_train_x_folds, il_train_y_folds, il_test_x_folds, il_test_y_folds):
    error_array = []
    for i in (1, 2, 3, 4, 5):
        error = calculate_fold_error(il_train_x_folds,il_train_y_folds, il_test_x_folds, il_test_y_folds,i)
        error_array = np.append(error_array, error)
    return error_array
    
# Calulate out of sample error for all k
error = k_error(il_train_x_folds, il_train_y_folds, il_test_x_folds, il_test_y_folds)

# Create table with out of sample errors
print("")
print("[Best model for Indoor Localization data]")
print("Table with out of sample errors for k = 1,2,3,4,5:")
t8 = Table()
t8['k value'] = [1, 2, 3, 4, 5]
t8['average error'] = error
print(t8)

# Choose the model with lowest estimated out of sample error
min_error = t8['average error'].min()
minimum = t8.where('average error', min_error)
min_k = int(minimum['k value'])
print("For the model with the lowest estimated out of sample error k equals: " + str(min_k))

# train best model with the full training set
predicted_y = knn_prediction(il_train_x,il_train_y,il_test_x,min_k)

# Kaggleize predictions
file_name = '../Predictions/IndoorLocalization/best.csv'
kaggle.kaggleize(predicted_y, file_name)




