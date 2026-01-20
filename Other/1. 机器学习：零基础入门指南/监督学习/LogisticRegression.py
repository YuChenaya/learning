# 逻辑回归：解决二分类问题
# 核心思想：计算某件事发生的概率（0-1之间）
# 实例应用：判断邮件是否为垃圾邮件

from sklearn.linear_model import LogisticRegression

# 特征
# feature1: 邮件包含"免费"次数
# feature2: 邮件包含"获奖"次数
x_train = [[3, 1], [5, 2], [1, 0], [0, 1]]  # 训练数据
Y_train = [1, 1, 0, 0]  # 1=垃圾邮件，0=正常邮件s

# 创建模型
model = LogisticRegression()
model.fit(x_train, Y_train)

# 预测
new_email = [[2, 2], [0, 0], [0, 2], [4, 3]]
prediction = model.predict(new_email)
print(["垃圾邮件" if p == 1 else "正常邮件" for p in prediction])