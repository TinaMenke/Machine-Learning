# Import python modules
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def read_D1():
	print('Reading D1 ...')
	D1 = np.load('../../Data/D1.npy')
	return D1

def read_D2():
	print('Reading D2 ...')
	D2 = np.load('../../Data/D2.npy')
	return D2

def read_image_data():
	print('Reading train data ...')
	temp = np.load('../../Data/data_train.npz')
	train_x = temp['data_train']
	temp = np.load('../../Data/labels_train.npz')
	train_y = temp['labels_train']
	print('The shape of the training data is '
		  '{:d} samples by {:d} features'.format(*train_x.shape))
	train_y = np.equal(train_y, 2*np.ones(train_y.shape)).astype(int)
	train_x = train_x[0:10000,:]
	train_y = train_y[0:10000]
	return (train_x, train_y)


def read_test_data():
	print('Reading test data ...')
	temp = np.load('../../Data/data_test.npz')
	test_x = temp['data_test']
	temp = np.load('../../Data/labels_test.npz')
	test_y = temp['labels_test']
	print('The shape of the test data is '
		  '{:d} samples by {:d} features'.format(*test_x.shape))
	test_y = np.equal(test_y, 2*np.ones(test_y.shape)).astype(int)
	test_x = test_x[0:1000,:]
	test_y = test_y[0:1000]
	return (test_x, test_y)

############## Question 3 ##################

D1 = read_D1()
D2 = read_D2()

#####################################################

# plotting configuration
fig, axs = plt.subplots(2, 2, figsize=(13,9), sharey=False)

# mean D1
mean_d1 = np.mean(D1)
print("mean D1: " + "{:,}".format(mean_d1))
# variance D1
variance_d1 = np.var(D1)
print("variance D1: " + "{:,}".format(variance_d1))
#plot D1
#bins = [0,20000,40000,60000,80000,100000,120000,140000,160000,180000,200000,220000,240000,260000,280000,300000]
axs[0,0].hist(D1, bins="auto")
axs[0,0].set_title("Histogram for D1 data")


# mean D2
mean_d2 = np.mean(D2)
print("mean D2: " + "{:,}".format(mean_d2))
#variance D2
variance_d2 = np.var(D2)
print("variance D2: " + "{:,}".format(variance_d2))
# plot D2
#bins = [0,20000,40000,60000,80000,100000,120000,140000,160000,180000,200000,220000,240000,260000,280000,300000]
axs[0,1].hist(D2, bins = "auto")
axs[0,1].set_title("Histogram for D2 data")


#bootstrapping D1
means = []
B = 10000
print("Number of bootstrap samples for D1: " +str(B))
for i in np.arange(B):
	sample = np.random.choice(D1, size=len(D1), replace=True)
	mean = np.mean(sample)
	means = np.append(means, mean)

left_end = np.percentile(means, 2.5)
right_end = np.percentile(means, 97.5)
print("95% confidence interval for D1: [" + "{:,}".format(left_end) + " ; " + "{:,}".format(right_end) + "]")

#histogram of the bootstrap means D1
axs[1,0].hist(means)
axs[1,0].set_title("Histogram of bootstrap means for D1")
# calculate variance means D1
mean_var_d1 = np.var(means)
print("variance means D1: " + "{:,}".format(mean_var_d1))

#bootstrapping D2
means = []
B = 10000
print("Number of bootstrap samples for D2: " +str(B))
for i in np.arange(B):
	sample = np.random.choice(D2, size=len(D2), replace=True)
	mean = np.mean(sample)
	means = np.append(means, mean)

left_end = np.percentile(means, 2.5)
right_end = np.percentile(means, 97.5)
print("95% confidence interval for D2: [" + "{:,}".format(left_end) + " ; " + "{:,}".format(right_end) + "]")

#histogram of the bootstrap means D2
axs[1,1].hist(means, bins = "auto")
axs[1,1].set_title("Histogram of bootstrap means for D2")
# calculate variance means D2
mean_var_d2 = np.var(means)
print("variance means D2: " + "{:,}".format(mean_var_d2))

plt.show()
fig.savefig('../Figures/plot_task_3.png')
############## Question 4  ##################

train_x, train_y = read_image_data()
test_x, test_y = read_test_data()

#####################################################

#Logistic Regression classifier
clf_logreg = LogisticRegression(random_state=7).fit(train_x, train_y)
y_pred = clf_logreg.predict(test_x)

#accuracy of Logistic Regression classifier
acc_log_reg = accuracy_score(test_y, y_pred)
print("The accuracy score of the logistic regression classifier is: " + str(acc_log_reg))

#SVM classifier
clf_svm = SVC(random_state=7)
clf_svm.fit(train_x, train_y)
y_pred = clf_svm.predict(test_x)


#accuracy of the SVM classifier
acc_svm = accuracy_score(test_y, y_pred, normalize=True)
print("The accuracy score of the SVM classifier is: " + str(acc_svm))

#calculate the difference in performance
performance_diff = acc_log_reg - acc_svm
print("The point estimate of their performance difference is: " + str(performance_diff))

#bootstrapping
accurancies_reg = []
accurancies_svm = []
accuracy_diff = []
B = 100
for i in np.arange(B):
	sample = np.random.choice(np.arange(len(test_x)), size=len(test_x), replace=True)
	sample_x = test_x[sample]
	sample_y = test_y[sample]
	y_pred_logreg = clf_logreg.predict(sample_x)
	y_pred_svm = clf_svm.predict(sample_x)
	acc_logreg = accuracy_score(sample_y, y_pred_logreg)
	acc_svm = accuracy_score(sample_y, y_pred_svm)
	accurancies_reg = np.append(accurancies_reg, acc_logreg)
	accurancies_svm = np.append(accurancies_svm, acc_svm)
	acc_difference = acc_logreg - acc_svm
	accuracy_diff = np.append(accuracy_diff, acc_difference)


#plot histogram of the empirical bootstrap distribution
#bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
plt.hist(accuracy_diff, bins = "auto")
plt.title("Histogram of the empirical bootstrap distribution")
plt.savefig('../Figures/plot_task_4.png')
plt.show()

#calculate confidence interval
left_end = np.percentile(accuracy_diff, 2.5)
right_end = np.percentile(accuracy_diff, 97.5)
print("95% confidence interval of empirical bootstrap distribution: [" + str(left_end) + " ; " + str(right_end) + "]")
















