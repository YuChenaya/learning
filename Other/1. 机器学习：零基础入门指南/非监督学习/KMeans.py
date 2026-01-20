# 聚类分析：物以类聚（K-means）
# 工作原理：自动将数据分成K个簇
# 实例应用：市场细分分析

import numpy as np
from sklearn.cluster import KMeans

# 假设有两种客户特征：购买频率和平均客单价
customer_data = np.array([
    [1, 100],  # 客户1
    [5, 500],  # 客户2
    [1, 150],  # 客户3
    [6, 550]  # 客户4
])

# 创建k=2的聚类模型
kmeans = KMeans(n_clusters=2)
kmeans.fit(customer_data)

print(kmeans.labels_)  # 输出每个客户的聚类标签