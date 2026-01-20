# 简单线性回归：预测连续值
# 核心思想：找到一条最佳拟合线，预测连续值结果
# 实例应用：根据房子面积预测房价

import numpy as np
from sklearn.linear_model import LinearRegression

# 房屋面积数据
x = np.array([50, 70, 90, 110])
house_areas = x.reshape(-1, 1)

# 房价
Y = np.array([300, 400, 500, 600])
prices = Y

# 创建模型（线性）
model = LinearRegression()
model.fit(house_areas, prices)

# 预测120平房子价格
prediction = model.predict(np.array([120, 100]).reshape(-1, 1))
print(prediction)
