## Learning Scikit-learn: Machine Learning in Python
## Raul Garreta & Guillermo Mocecchi

## IPython Notebook for Chapter 3: Unsupervised Learning - Clustering Handwritten Digits

# Clustering involves finding groups where all elements in the group are similar, but objects in different groups are not.
# K-means is the most popular clustering algorithm, because it is very simple and easy to implement and
# it has shown good performance on different tasks. We will show in this notebook how k-means works using a
# motivating example, the problem of clustering handwritten digits. At the end of the notebook, we will try other,
# different, clustering approaches to the same problem.


# Activate your Virtual Environment: `source bin/activate`
# Install the libraries necessary for this chapter: `pip3 install numpy scikit-learn matplotlib`
# Then start up an interactive shell which is matplotlib aware: `ipython --pylab`

# Now import numpy, scikit-learn, and pyplot, the Python libraries we will be using in this chapter.
# Show the versions we will be using (in case you have problems running the notebooks).

import sklearn as sk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print('numpy version:', np.__version__)
print('scikit-learn version:', sk.__version__)
print('matplotlib version:', matplotlib.__version__)


# MNIST handwritten digits dataset
# 43 people handwrote 10 digits 0..9. 30 people contributed to the training set and a different 13 to the test set.
# 32x32 bitmaps are divided into non-overlapping blocks of 4x4 and the number of 'on' pixels are counted in each block.
# This generates an input matrix of 8x8 where each element is an integer in the range 0..16

# Import the dataset (http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html)
# and show some of its instances

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
digits = load_digits()

# try:
digits.data
digits.target[0:20]

# normalisation: a common requirement for many machine learning estimators implemented in scikit-learn;
# they might behave badly if the individual features do not more or less look like standard normally distributed data
# a Gaussian distribution with zero mean and unit variance (st.dev=1).
data = scale(digits.data)

# try:
data

def print_digits(images,y,max_n=10):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    i = 0
    while i < max_n and i < images.shape[0]:
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the image with the target value
        p.text(0, 14, str(y[i]))
        i = i + 1
    
print_digits(digits.images, digits.target, max_n=10)


# Build training and test set
from sklearn.model_selection import train_test_split

# Since from scikit-learn 0.15.1 train_test_split only admits dim=2 arrays
# We have to reshape images
number_of_instances=digits.images.shape[0]
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        data, digits.target, digits.images.reshape(number_of_instances,64),test_size=0.25, random_state=42)

# try:
digits.images.shape
digits.images[0]
images_train[0]

# try:
test1 = [1, 3, 5, 7]
train_test_split(test1, test_size=0.25)
test2 = [2, 4, 6, 8]
train_test_split(test1, test2, test_size=0.25)
train_test_split(test1, test2, test_size=0.25)
train_test_split(test1, test2, test_size=0.25, random_state=99)
train_test_split(test1, test2, test_size=0.25, random_state=99)

# continue:
n_samples, n_features = X_train.shape
n_digits = len(np.unique(y_train))
labels = y_train

# try:
X_train.shape
np.unique(y_train)
np.unique(y_test)
labels

# Reshape images back
images_train[0]
images_train=images_train.reshape(images_train.shape[0],8,8)
images_test=images_test.reshape(images_test.shape[0],8,8)
images_train[0]

# sprintf
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


## K-Means
# The main idea behind k-means is to find a partition of data points such that the squared distance between
# the cluster mean and each point in the cluster is minimized. Note that this method assumes that you know
# a priori the number of clusters your data should be divided into.
# The number of clusters (k) is user-defined. Each cluster is described by a single point known as the centroid.
# A centroid is at the centre of all the points in a cluster.
# First the k centroids are randomly scattered around the dataset.
# Each dataset instance is then assigned to a cluster by finding the closest centroid.
# The centroids are updated at every step by recalculating the mean value of all the points in each cluster.

# Train a K-Means classifier, show the clusters. 
from sklearn import cluster
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)
clf.fit(X_train)
print(clf.labels_.shape)
print(clf.labels_[0:9])
print_digits(images_train, clf.labels_, max_n=10)

# To predict the clusters for new data, we use the usual predict method of the classifier.
y_pred = clf.predict(X_test)

def print_cluster(images, y_pred, cluster_number):
    images = images[y_pred==cluster_number]
    y_pred = y_pred[y_pred==cluster_number]
    print_digits(images, y_pred, max_n=10)

for i in range(10):
     print_cluster(images_test, y_pred, i)

# How can we evaluate our performance? Precision and all that stuff does not work, since
# we have no target classes to compare with. To evaluate, we need to know the "real" clusters,
# whatever that means. We can suppose, for our example, that each cluster includes every
# drawing of a certain number, and only that number. Knowing this, we can compute the adjusted
# Rand index between our cluster assignment and the expected one. The Rand index is a similar
# measure for accuracy, but it takes into account the fact that classes can have different
# names in both assignments. That is, if we change class names, the index does not change.
# The adjusted index tries to deduct from the result coincidences that have occurred by
# chance. When you have the exact same clusters in both sets, the Rand index equals one,
# while it equals zero when there are no clusters sharing a data point. Show different
# performance metrics, compared with "original" clusters (using the known number class)

from sklearn import metrics
print("Adjusted rand score:{:.2}".format(metrics.adjusted_rand_score(y_test, y_pred)))
print("Homogeneity score:{:.2} ".format(metrics.homogeneity_score(y_test, y_pred)) )
print("Completeness score: {:.2} ".format(metrics.completeness_score(y_test, y_pred)))
print("Confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred))

# Draw clusters and centroids (taken from the scikit-learn tutorial
# http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

from sklearn import decomposition

# start with principal components analysis for dimensionality reduction
pca = decomposition.PCA(n_components=2).fit(X_train)
reduced_X_train = pca.transform(X_train)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .01     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will asign a colour to each
x_min, x_max = reduced_X_train[:, 0].min() + 1, reduced_X_train[:, 0].max() - 1
y_min, y_max = reduced_X_train[:, 1].min() + 1, reduced_X_train[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# In this case the seeding of the centres is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
kmeans = cluster.KMeans(init='k-means++', n_clusters=n_digits, n_init=1)
kmeans.fit(reduced_X_train)
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the centroids as white dots
centroids = kmeans.cluster_centers_

# Put the result into a colour plot
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap=plt.cm.Paired,
          aspect='auto', origin='lower')
plt.plot(reduced_X_train[:, 0], reduced_X_train[:, 1], 'k.', markersize=2)
plt.scatter(centroids[:, 0], centroids[:, 1],
           marker='.', s=169, linewidths=3,
           color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
         'Centroids are marked with white dots')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


## Affinity Propagation

# A typical problem for clustering is that most methods require the number of clusters we want to identify.
# The general approach to solve this is to try different numbers and let an expert determine which works
# best using techniques such as dimensionality reduction to visualize clusters. There are also some methods
# that try to automatically calculate the number of clusters. Scikit-learn includes an implementation of
# Affinity Propagation, a method that looks for instances that are the most representative of others,
# and uses them to describe the clusters. Since scikit-learn uses the same function for every algorithm,
# we just have to fit the training set again: Try now using Affinity Propagation

aff = cluster.AffinityPropagation()
aff.fit(X_train)
print(aff.cluster_centers_indices_.shape)
# print_digits(images_train[aff.cluster_centers_indices_], y_train[aff.cluster_centers_indices_], max_n=aff.cluster_centers_indices_.shape[0])


## Mixture of Gaussian Models

# Finally, we will try a probabilistic approach to clustering, using Gaussian Mixture Models (GMM). We
# will see, from a procedural view, that it is very similar to k-means, but their theoretical principles
# are quite different. GMM assumes that data comes from a mixture of finite Gaussian distributions with
# unknown parameters.

from sklearn import mixture

# Which covariance type to use?
for covariance_type in ['spherical','tied','diag','full']:
    gm=mixture.GaussianMixture(n_components=n_digits, covariance_type=covariance_type, random_state=42, n_init=5)
    gm.fit(X_train)
    y_pred=gm.predict(X_test)
    print("Adjusted rand score for covariance={}:{:.2}".format(covariance_type, metrics.adjusted_rand_score(y_test, y_pred)))

# Which one performed best?
bestCovariance='spherical'

# Training
gm = mixture.GaussianMixture(n_components=n_digits, covariance_type=bestCovariance, random_state=42)
gm.fit(X_train)

# Predict and evaluate
y_pred = gm.predict(X_test)
print("Adjusted rand score:{:.2}".format(metrics.adjusted_rand_score(y_test, y_pred)))
print("Homogeneity score:{:.2} ".format(metrics.homogeneity_score(y_test, y_pred)))
print("Completeness score: {:.2} ".format(metrics.completeness_score(y_test, y_pred)))

# Print image clusters and confusion matrix
for i in range(10):
     print_cluster(images_test, y_pred, i)
print("Confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred))
