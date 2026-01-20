# 降维技术：化繁为简
# 工作原理：将高维数据压缩到关键维度
# 实例应用：人脸识别特征提取

# PCA降维示例
from sklearn.decomposition import PCA
import numpy as np

# 创建一些三维数据
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# 创建PCA模型，降到二维
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

print("降维后数据:")
print(reduced_data)