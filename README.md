# Machine-Learning
A variety of small machine learning projects.

In the following, the task of each machine learning project is briefly described. Unfortunately, more detailed information may not be published. 

## Regression

Train a regression model of your choice from either decision trees, nearest neighbors or linear models
on the power plant dataset. Pick ranges of hyperparameters that you would like to experiment
with (depth for decision trees, number of neighbors for nearest neighbors and regularization constants
for linear models). Your task is to make predictions on the test set, kagglize your output and submit
to kaggle public leadership score. Repeat the previous question with the indoor localization dataset

## Classification

You are given a set of gray scale images (32 x 32 pixels) with one (and only one) of the following
objects: horse, truck, frog, ship (labels 0, 1, 2 and 3, respectively). The goal is to train a model to recognize
which of the objects is present in an image. You will train different models (Decision Trees, Nearest Neighbors, Linear models, Neural Networks), compare
their performances and training time, and perform model selection using cross-validation.
The data set provided is the training set: 20; 000 samples with 1024 features each (pixel values, in this case).
We will have a separate test set that we will run your code against in Gradescope.

## Kernel-Ridge Regression

In this homework you will experimenting with kernel methods for regression and classification problems
using kernel ridge regression and support vector machine (SVM).

## Probabilistic Methods

You’re trying to estimate the average salary of people in San Francisco. If you gathered 500 datapoints, your
estimate might not be as reliable as if you gathered 10,000 datapoints. We have supplied two such datasets
simulating this scenario (we actually sampled them from a real complete dataset), with these two different
sizes. Implement bootstrapping for the mean, using the percentile-based confidence interval method. Use
at least B = 100 bootstrap samples and say how many you’re using. (Also save the bootstrap means for
the next part.) For each dataset, report the 95% confidence interval. 

You have been given a image classification dataset with two output classes. There size of the training set
is 10,000 and size of the test set is 1,000. You are required to train 2 classifiers i.e. a Logistic Regression
classifier and a Support Vector Machine classifier to perform classification on this data.

## Clustering

Unsupervised learning: Contrary to supervised learning (classification, regression), unsupervised learning
algorithms learn patterns from unlabeled examples. There are several popular algorithms in unsupervised
learning such as principal component analysis (PCA), k-means, independent component analysis (ICA) and
density estimation. In this project you will use k-means to compress images.
