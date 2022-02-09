import scipy
import torch
import numpy as np
from kmeans_pytorch import kmeans,  kmeans_predict
import a_data as ad


from scipy.cluster.vq import vq, kmeans, whiten





# data
# data_size, dims, num_clusters = 1000, 2, 3
# x = np.random.randn(data_size, dims) / 6
# x = torch.from_numpy(x)

from sklearn.cluster import KMeans
df = ad.training_data.data
kmeans1 = KMeans(n_clusters=ad.training_data_len-1, init='k-means++', max_iter=500, n_init=10)
y_pred = kmeans1.fit_predict(df)
centroids = kmeans1.cluster_centers_
print(y_pred)
sum = 0
for i in range(len(y_pred)):
    sum += np.linalg.norm(np.array(centroids[y_pred[i]]) - np.array(df[i]))
print(sum/len(y_pred))
# data_size = ad.training_data_len
# dims = 50#ad.wandb.config["embedding_dim"]
# num_clusters = ad.training_data_len//10 #ad.wandb.config["num_embeddings"]
# x= ad.training_data.data
# #df = pd.DataFrame(np.random.randn(1000, 5), columns=list('XYZVW'))
# centers, loss = kmeans(x, num_clusters)
# print(loss)
# #df['Cluster'] = vq(features, centers)[0]




# x= torch.from_numpy(x)
#
#
# # kmeans
# _, cluster_centers = kmeans(
#     X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
# )
#
# loss = 0
# for i in range(ad.training_data_len):
#     y = next(iter(ad.training_loader))
#     y = y.data.reshape(1, 50)
#
#     k_mean = kmeans_predict(y, cluster_centers, 'euclidean',
#                                     device=torch.device('cuda:0'))
#     loss =+ torch.cdist(y, k_mean,p=2)
# print(loss/ad.training_data_len)

# print(cluster_centers)
# y = np.random.randn(5, dims) / 6
# y = torch.from_numpy(y)
#
# cluster_ids_y = kmeans.kmeans_predict(
#     y, cluster_centers, 'euclidean', device=torch.device('cuda:0')
# )
#
# print(cluster_ids_y)

#
# plt.figure(figsize=(8, 8))
# plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), c=cluster_ids_x.cpu(), s=30000 / len(x), cmap="tab10")
# plt.scatter(cluster_centers[:, 0].cpu(), cluster_centers[:, 1].cpu(), c="black", s=50, alpha=0.8)
# plt.axis([-2, 2, -2, 2])
# plt.tight_layout()
# plt.show()