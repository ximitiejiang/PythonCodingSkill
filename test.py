from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成样本X(200,2), 标签y(200,)
X, y = make_blobs(n_samples=200,
                  n_features=2,
                  cluster_std=1,
                  center_box=(-10., 10.),
                  shuffle=True,
                  random_state=1)

plt.figure(figsize=(6,4), dpi=144)
plt.scatter(X[:,0], X[:,1])

# 创建算法，拟合数据
kmean = KMeans(n_clusters=3)
kmean.fit(X)
# 查看结果
score = kmean.score(X)             # float64
centers = kmean.cluster_centers_   # (3,2)
preds = kmean.labels_              # (200,)

# 显示
plt.scatter(centers[:,0], centers[:,1], s=50, c='b')