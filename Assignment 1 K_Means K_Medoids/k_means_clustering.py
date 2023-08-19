import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import plotly.express as px
from sklearn.preprocessing import StandardScaler
# %matplotlib inline
plt.style.use('dark_background')

#data = pd.read_csv('C:/Users/intaj/Desktop/2211MC09/cancer.csv') # local directory path of cancer.csv
from google.colab import drive
drive.mount('/content/drive')

"""## Reading the Country data CSV file """

data= pd.read_csv('/content/drive/MyDrive/ML CSV/cancer.csv')
## Displaying the data
data.head()

data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, inplace=True) # Removing the data columns which need to drop as per assignment.

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data) # Scaling the dataset to reduce dataset variance
# statistics of scaled data
pd.DataFrame(data_scaled).describe()

class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters # cluster number
        self.max_iterations = 100 # max iteration. don't want to run inf time
        self.num_examples, self.num_features = X.shape # num of examples, num of features
        self.plot_figure = True # plot figure

    # randomly initialize centroids
    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features)) # row , column full with zero
        for k in range(self.K): # iterations of
            centroid = X[np.random.choice(range(self.num_examples))] # random centroids
            centroids[k] = centroid
        return centroids # return random centroids

    # create cluster Function
    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.K)]
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point-centroids)**2, axis=1))
            ) # closest centroid using euler distance equation(calculate distance of every point from centroid)
            clusters[closest_centroid].append(point_idx)
        return clusters

    # new centroids
    def calculate_new_centroids(self, cluster, X):
        centroids = np.zeros((self.K, self.num_features)) # row , column full with zero
        for idx, cluster in enumerate(cluster):
            new_centroid = np.mean(X[cluster], axis=0) # find the value for new centroids
            centroids[idx] = new_centroid
        return centroids

    # prediction
    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples) # row1 fillup with zero
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred

    # plotinng scatter plot
    def plot_fig(self, X, y):
        fig = px.scatter(X[:, 0], X[:, 1], color=y)
        fig.show() # visualize


    # fit data
    def fit(self, X):
        centroids = self.initialize_random_centroids(X) # initialize random centroids
        for _ in range(self.max_iterations):
            clusters = self.create_cluster(X, centroids) # create cluster
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X) # calculate new centroids
            diff = centroids - previous_centroids # calculate difference
            if not diff.any():
                break
        y_pred = self.predict_cluster(clusters, X) # predict function
        if self.plot_figure: # if true
            self.plot_fig(X, y_pred) # plot function
        return y_pred

if __name__ == "__main__":
   np.random.seed(10)
   num_clusters = 2 # num of cluster
   X = np.array(data_scaled.astype(float))
  #  print(data.shape)
   Kmeans = KMeansClustering(X, num_clusters)
   y_pred = Kmeans.fit(X)
  #  print(y_pred)

print(y_pred) # Printing the predictions of cluster points

frame = pd.DataFrame(data)
frame['cluster'] = y_pred
frame['cluster'].value_counts() # Printing the cluster points in each cluster.

dataset = data.copy()
dataset['cluster'] = y_pred

fig = px.scatter_3d(dataset, x="radius_mean", y="texture_mean", z="perimeter_mean", color='cluster', size_max=30)
fig.show() # ploting 3D scatter plot for better visualizing of cluster points.

fig = px.scatter(dataset['radius_mean'], dataset['texture_mean'], color=dataset['cluster'])
fig.show() # Ploting scatter plot using radius_mean and texture_mean feature vector.
