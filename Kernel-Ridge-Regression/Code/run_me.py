# Import python modules
import numpy as np
import kaggle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.svm import SVC

############################################################################
# Read in train and test synthetic data
def read_synthetic_data():
	print('Reading synthetic data ...')
	train_x = np.loadtxt('../../Data/Synthetic/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/Synthetic/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/Synthetic/data_test.txt', delimiter = ',', dtype=float)
	test_y = np.loadtxt('../../Data/Synthetic/label_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x, test_y)

############################################################################
# Read in train and test credit card data
def read_creditcard_data():
	print('Reading credit card data ...')
	train_x = np.loadtxt('../../Data/CreditCard/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/CreditCard/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/CreditCard/data_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x)

############################################################################
# Read in train and test tumor data
def read_tumor_data():
	print('Reading tumor data ...')
	train_x = np.loadtxt('../../Data/Tumor/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/Tumor/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/Tumor/data_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x)

############################################################################
# Compute MSE
def compute_MSE(y, y_hat):
	# mean squared error
	return np.mean(np.power(y - y_hat, 2))

############################################################################

train_x, train_y, test_x, test_y = read_synthetic_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

############################################################################
#task 2d

#reshaping the data
train_x = np.reshape(train_x,[200,1])
train_y = np.reshape(train_y,[200,1])
test_x = np.reshape(test_x,[200,1])
test_y = np.reshape(test_y,[200,1])

#Defining the polynomial KRRS function
def KRRS_poly(x1,x2,i):
	return (1+x1 * x2)**i

#calculate kernel matrix
K = np.empty([200,200])
N = 200
y_poly_pred = np.empty([200,1])
orders = [1,2,4,6]
predictions_poly = []
mse_poly = []

for i in orders:
	# Reset the kernel for every run of the order
	K = np.zeros_like(K)
	y_poly_pred = np.zeros_like(y_poly_pred)
	
	# Run trhough the Kernel K of shape NxN. Each (j,k)'th element of the
	# kernel is equal to either th epolynomial or trigonometric method of 
	# the kernel, seen in the respective functions.
	for j in range(N):
		for k in range(N):
			K[j][k] = KRRS_poly(train_x[j], train_x[k],i)
	
	# alpha is of shape (Nx1)
	alpha = np.linalg.solve(K+np.diag(0.1*np.ones(200)), train_y)
	
	# Now go through the kernel again, same thing as before, but use the values
	# from test data and the values calculated in alpha to make predictions
	for j in range(N):
		summation = 0
		for k in range(N):
			summation += alpha[k]*KRRS_poly(test_x[j],train_x[k],i)
		y_poly_pred[j] = summation
	# Append the prediction list, this will save all the data for plotting
	predictions_poly.append(y_poly_pred)
	mse_poly.append(compute_MSE(test_y, y_poly_pred))
print("MSE values for polynomial KRRS function:")
print(mse_poly)

#basis expansion faunction for polynomial function
def basis_expansion_poly(i,x):
	x_expanded = []
	for a in np.arange(0,i+1):
		x_expanded = np.append(x_expanded, x**a)
	return x_expanded

#basis expansion for polynomial function
orders = [1,2,4,6]
train_x_expansion = []
train_y_expansion = []
test_x_expansion = []
for i in orders:
	#basis expansion for train_x
	train_x_exp = []
	for j in np.arange(len(train_x)):
		train_x_exp.append(basis_expansion_poly(i,train_x[j]))
	train_x_exp = np.array(train_x_exp)
	train_x_expansion.append(train_x_exp)
		
	#basis expansion for test_x
	test_x_exp = []
	for j in np.arange(len(test_x)):
		test_x_exp.append(basis_expansion_poly(i,test_x[j]))
	test_x_exp = np.array(test_x_exp)
	test_x_expansion.append(test_x_exp)

#BERR function for polynomial order
pred_poly_BERR = []
clf = Ridge(alpha=0.1)
for i in np.arange(4):
	clf.fit(train_x_expansion[i], train_y)
	prediction = clf.predict(test_x_expansion[i])
	pred_poly_BERR.append(prediction)

# calculate MSE
mse_poly_BERR = []
mse_poly_BERR.append(compute_MSE(test_y, pred_poly_BERR[0]))
mse_poly_BERR.append(compute_MSE(test_y, pred_poly_BERR[1]))
mse_poly_BERR.append(compute_MSE(test_y, pred_poly_BERR[2]))
mse_poly_BERR.append(compute_MSE(test_y, pred_poly_BERR[3]))
print("MSE values for polynomail BERR function:")
print(mse_poly_BERR)

#Definition of trigonometric KRRS function
def KRRS_trig(x1,x2,i):
	results = 0
	for l in np.arange(1,i+1):
		result = np.sin(l*0.5*x1)*np.sin(l*0.5*x2)+np.cos(l*0.5*x1)*np.cos(l*0.5*x2)
		results = results+result
	return 1+results

#calculate kernel matrix
y_trig_pred = np.empty([200,1])
orders = [3,5,10]
predictions_trig = []
mse_trig = []

for i in orders:
	# Reset the kernel for every run of the order
	K = np.zeros_like(K)
	y_trig_pred = np.zeros_like(y_trig_pred)
	
	# Run trhough the Kernel K of shape NxN. Each (j,k)'th element of the
	# kernel is equal to either th epolynomial or trigonometric method of 
	# the kernel, seen in the respective functions.
	for j in range(N):
		for k in range(N):
			K[j][k] = KRRS_trig(train_x[j], train_x[k],i)
			
	alpha = np.linalg.solve(K+np.diag(0.1*np.ones(200)), train_y)
	
	# Now go through the kernel again, same thing as before, but use the values
	# from test data and the values calculated in alpha to make predictions
	for j in range(N):
		summation = 0
		for k in range(N):
			summation += alpha[k]*KRRS_trig(test_x[j],train_x[k],i)
		y_trig_pred[j] = summation
	# Append the prediction list, this will save all the data for plotting
	predictions_trig.append(y_trig_pred)
	mse_trig.append(compute_MSE(test_y, y_trig_pred))
print("MSE values for trigonometric KRRS function:")
print(mse_trig)

#basis expansion function for trigonometric order
def basis_expansion_trig(i,x):
	x_expanded = [1]
	for a in np.arange(1,i+1):
		x_expanded = np.append(x_expanded, np.sin((a)*0.5*x))
		x_expanded = np.append(x_expanded, np.cos((a)*0.5*x))
	return x_expanded

orders = [3,5,10]
train_x_expansion = []
test_x_expansion = []
for i in orders:
	#basis expansion for train_x
	train_x_exp = []
	for j in np.arange(len(train_x)):
		train_x_exp.append(basis_expansion_trig(i,train_x[j]))
	train_x_exp = np.array(train_x_exp)
	train_x_expansion.append(train_x_exp)
		
	#basis expansion for test_x
	test_x_exp = []
	for j in np.arange(len(test_x)):
		test_x_exp.append(basis_expansion_trig(i,test_x[j]))
	test_x_exp = np.array(test_x_exp)
	test_x_expansion.append(test_x_exp)

#BERR function for trigonometric order
pred_trig_BERR = []
clf = Ridge(alpha=0.1)
for i in np.arange(3):
	clf.fit(train_x_expansion[i], train_y)
	prediction = clf.predict(test_x_expansion[i])
	pred_trig_BERR.append(prediction)

# calculate MSE
mse_trig_BERR = []
mse_trig_BERR.append(compute_MSE(test_y, pred_trig_BERR[0]))
mse_trig_BERR.append(compute_MSE(test_y, pred_trig_BERR[1]))
mse_trig_BERR.append(compute_MSE(test_y, pred_trig_BERR[2]))
print("MSE values for polynomail BERR function:")
print(mse_trig_BERR)

#Plotting
fig, axs = plt.subplots(4, 2, figsize=(15, 25), sharey=True)
#first row, first column
axs[0,0].set_title('KRRS, Polynomial, degree = 2, labda = 0.1')
axs[0,0].set_ylabel('True/Predicted Y')
axs[0,0].set_xlabel('Test X')
axs[0,0].scatter(test_x, predictions_poly[1], c="red", marker='o')
axs[0,0].scatter(test_x, test_y, c="blue", marker='*')
#first row, second column
axs[0,1].set_title('BERR, Polynomial, degree = 2, labda = 0.1')
axs[0,1].set_ylabel('True/Predicted Y')
axs[0,1].set_xlabel('Test X')
axs[0,1].scatter(test_x, pred_poly_BERR[1], c="red", marker='o')
axs[0,1].scatter(test_x, test_y, c="blue", marker='*')
#second row, first column
axs[1,0].set_title('KRRS, Polynomial, degree = 6, labda = 0.1')
axs[1,0].set_ylabel('True/Predicted Y')
axs[1,0].set_xlabel('Test X')
axs[1,0].scatter(test_x, predictions_poly[3], c="red", marker='o')
axs[1,0].scatter(test_x, test_y, c="blue", marker='*')
#second row, second column
axs[1,1].set_title('BERR, Polynomial, degree = 6, labda = 0.1')
axs[1,1].set_ylabel('True/Predicted Y')
axs[1,1].set_xlabel('Test X')
axs[1,1].scatter(test_x, pred_poly_BERR[3], c="red", marker='o')
axs[1,1].scatter(test_x, test_y, c="blue", marker='*')
#third row, first column
axs[2,0].set_title('KRRS, Trigonometric, degree = 5, labda = 0.1, delta = 0.5')
axs[2,0].set_ylabel('True/Predicted Y')
axs[2,0].set_xlabel('Test X')
axs[2,0].scatter(test_x, predictions_trig[1], c="red", marker='o')
axs[2,0].scatter(test_x, test_y, c="blue", marker='*')
#third row, second column
axs[2,1].set_title('BERR, Trigonometric, degree = 5, labda = 0.1, delta = 0.5')
axs[2,1].set_ylabel('True/Predicted Y')
axs[2,1].set_xlabel('Test X')
axs[2,1].scatter(test_x, pred_trig_BERR[1], c="red", marker='o')
axs[2,1].scatter(test_x, test_y, c="blue", marker='*')
#fourth row, first column
axs[3,0].set_title('KRRS, Trigonometric, degree = 10, labda = 0.1, delta = 0.5')
axs[3,0].set_ylabel('True/Predicted Y')
axs[3,0].set_xlabel('Test X')
axs[3,0].scatter(test_x, predictions_trig[2], c="red", marker='o')
axs[3,0].scatter(test_x, test_y, c="blue", marker='*')
#fourth row, second column 
axs[3,1].set_title('BERR, Trigonometric, degree = 10, labda = 0.1, delta = 0.5')
axs[3,1].set_ylabel('True/Predicted Y')
axs[3,1].set_xlabel('Test X')
axs[3,1].scatter(test_x, pred_trig_BERR[2], c="red", marker='o')
axs[3,1].scatter(test_x, test_y, c="blue", marker='*')
#plt.show()
fig.savefig('../Figures/plot_2e.png')

############################################################################
#task 2e

train_x, train_y, test_x  = read_creditcard_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

#cross validation function
def cross_validation(train_x, train_y, k=7):
	kf = KFold(n_splits=k, shuffle=True, random_state=7)
	kf.get_n_splits(train_x)
	# arrays to save testing and training data of all five folds
	kf_X_train, kf_X_test, kf_y_train, kf_y_test = [], [], [], []

	for train_index, test_index in kf.split(train_x):
		X_train, X_test = train_x[train_index], train_x[test_index]
		y_train, y_test = train_y[train_index], train_y[test_index]

		kf_X_train.append(X_train)
		kf_X_test.append(X_test)
		kf_y_train.append(y_train)
		kf_y_test.append(y_test)

	return (kf_X_train, kf_X_test, kf_y_train, kf_y_test)

#Ridge Regression for Credit Card dataset
print("MSE for Ridge Regression on Credit Card dataset:")
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
X = train_x
y =  train_y
predictions = []
min_error = 100000
alpha = [1,0.0001]
models = ["linear", "poly", "rbf"]
gamma = [None,1,0.001]
for m in models:
	for a in alpha:
		for g in gamma:
			errors = []
			clf = KernelRidge(alpha=a,kernel = m, gamma=g)
			x_train, x_test, y_train, y_test = cross_validation(X, y, k=5)
			for i in np.arange(5):
				clf.fit(x_train[i], y_train[i])
				prediction = clf.predict(x_test[i])
				error = compute_MSE(y_test[i], prediction)
				errors = np.append(errors, error)
			print(np.mean(errors))
			if(np.mean(errors)<min_error):
				min_error = np.mean(errors)
				best_model = clf

#try three best models by hand
#best_model = KernelRidge(alpha=0.0001,kernel = "poly", gamma=0.001)
#best_model = KernelRidge(alpha=0.0001,kernel = "rbf", gamma=0.001)
best_model = KernelRidge(alpha=0.0001,kernel = "rbf", gamma=None)
best_model.fit(X,y)

#print best model
print("The best ridge ression model is:")
print(best_model)

#predict on full test data 
train_x, train_y, test_x  = read_creditcard_data()
prediction = best_model.predict(test_x)

#kaggleize the data
file_name = '../Predictions/CreditCard/best.csv'
kaggle.kaggleize(prediction,file_name,True)

############################################################################
#task 3

#load data
train_x, train_y, test_x  = read_tumor_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

#make predictions using SVM
print("Accuracy for SVM on Tumor dataset:")
X = train_x
y =  train_y
predictions = []
models = ["rbf", "poly", "linear"]
degree = [3,5]
C = [1, 0.01, 0.0001]
gamma = [1, 0.01, 0.001]
max_accuracy = 0
for m in models:
	for d in degree:
		for c in C:
			for g in gamma:
				cross_accuracy = []
				clf = SVC(C= c, kernel = m, degree = d, gamma=g)
				x_train, x_test, y_train, y_test = cross_validation(X, y, k=5)
				for i in np.arange(5):
					clf.fit(x_train[i], y_train[i])
					prediction = clf.predict(x_test[i])
					acc = accuracy_score(y_test[i], prediction, normalize=True)
					cross_accuracy = np.append(cross_accuracy, acc)   
				print(np.mean(cross_accuracy))
				if(np.mean(cross_accuracy)>max_accuracy):
					max_accuracy = np.mean(cross_accuracy)
					best_model = clf
					
print("The best SVM model is:")
print(best_model)


#train on full trianing set and predict on full test data 
train_x, train_y, test_x  = read_tumor_data()
best_model.fit(train_x,train_y)
prediction = best_model.predict(test_x)

#kaggleize the data
file_name = '../Predictions/Tumor/best.csv'
kaggle.kaggleize(prediction,file_name,False)


############################################################################

# Create dummy test output values to compute MSE
#test_y = np.random.rand(test_x.shape[0], train_y.shape[1])
#predicted_y = np.random.rand(test_x.shape[0], train_y.shape[1])
#print('DUMMY MSE=%0.4f' % compute_MSE(test_y, predicted_y))

# Output file location
#file_name = '../Predictions/CreditCard/best.csv'
# Writing output in Kaggle format
#print('Writing output to ', file_name)
#kaggle.kaggleize(predicted_y, file_name, True)


# Create dummy test output values to compute accuracy
#test_y = np.random.randint(0, 2, (test_x.shape[0], 1))
#predicted_y = np.random.randint(0, 2, (test_x.shape[0], 1))
#print('DUMMY Accuracy=%0.4f' % accuracy_score(test_y, predicted_y, normalize=True))

# Output file location
#file_name = '../Predictions/Tumor/best.csv'
# Writing output in Kaggle format
#print('Writing output to ', file_name)
#kaggle.kaggleize(predicted_y, file_name, False)

