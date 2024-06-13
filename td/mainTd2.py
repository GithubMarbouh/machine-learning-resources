 # matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from scipy import ndimage
from sklearn import cluster
from sklearn.cluster import KMeans

image = imread("House.jpg")
plt.figure(figsize = (15,8))
plt.imshow(image)
plt.show()
image.shape
x, y, z = image.shape
image_2d = image.reshape(x*y, z)
image_2d.shape
kmeans_cluster = cluster.KMeans(n_clusters=10)
kmeans_cluster.fit(image_2d)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_centers
cluster_labels = kmeans_cluster.labels_
cluster_labels
plt.imshow((cluster_centers[cluster_labels].reshape(x, y, z) * 255).astype(np.uint8))
plt.show()
pic = plt.imread('nature.jpg') / 255
print(pic.shape)
plt.imshow(pic)
plt.show()
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
pic_n.shape
kmeans = KMeans(n_clusters=4, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]
cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)
plt.show()
print(pic.shape)
pic = plt.imread('grenouille.jpg') / 255
print(pic.shape)
plt.imshow(pic)
plt.show()
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
pic_n.shape
kmeans = KMeans(n_clusters=4, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]
cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)
plt.show()
image = imread("images.png")
plt.figure(figsize = (15,8))
plt.imshow(image)
plt.show()
image.shape
x, y, z = image.shape
image_2d = image.reshape(x*y, z)
image_2d.shape
kmeans_cluster = cluster.KMeans(n_clusters=2)
kmeans_cluster.fit(image_2d)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_centers
cluster_labels = kmeans_cluster.labels_
cluster_labels
plt.imshow((cluster_centers[cluster_labels].reshape(x, y, z) * 255).astype(np.uint8))
plt.show()
