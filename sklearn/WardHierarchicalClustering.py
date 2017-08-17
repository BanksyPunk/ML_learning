# 沃德结构层次聚类的浣熊脸图像的演示
# A demo of structured Ward hierarchical sklearn on a raccoon face image
import time as time

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version

if sp_version < (0, 12):
    raise SkipTest(
        "Skipping because SciPy version earlier than 0.12.0 and thus does not include the scipy.misc.face() image.")

# 创建数据
from scipy import misc
# face = misc.imread("demo.jpg", flatten=True)
face = misc.face(gray=True)

# Resize it to 10% of the original size to speed up the processing
face = sp.misc.imresize(face, 1.00) #/ 255.0  # divide 255

X = np.reshape(face, (-1, 1))

# Define the structure A of the data, Pixels connected to their neighbors
connectivity = grid_to_graph(*face.shape)

print("Compute structured hierarchical sklearn...")
st = time.time()
n_clusters = 15  # 区域的数量
# 此处传入的参数 connectivity 的作用是什么？不加入的话则运行时间要长得多，且图像效果也跟加入的差别很大
ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", connectivity=connectivity)
ward.fit(X)
label = np.reshape(ward.labels_, face.shape)
print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.size)
print("Number of clusters: ", np.unique(label).size)

# Plot the results on an image
plt.figure(figsize=(5, 5))
plt.imshow(face)
for cluster_num in range(n_clusters):
    plt.contour(label == cluster_num, contours=1, colors=[plt.cm.spectral(cluster_num / float(n_clusters)), ])
plt.xticks(())
plt.yticks(())
plt.show()