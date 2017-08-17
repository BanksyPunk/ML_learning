# 贝叶斯岭回归（Bayesian Ridge Regression）
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import BayesianRidge, LinearRegression

# 生成具有高斯权重模拟数据
np.random.seed(0)    # 设置 random 函数的随机种子
n_samples, n_features = 100, 100
X = np.random.randn(n_samples, n_features)    # 创建高斯数据集
lambda_ = 4.
w = np.zeros(n_features)
relevant_features = np.random.randint(0, n_features, 10)
print(relevant_features)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
print("w: ", w)
alpha_ = 50
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_), size=n_samples)
y = np.dot(X, w) + noise
print("X: ", X)
print("y: ", y)

# 拟合和比较 BayesianRidge 与 LinearRegression 方法
clf = BayesianRidge(compute_score=True)
clf.fit(X, y)
ols = LinearRegression()
ols.fit(X, y)

lw = 2
plt.figure(figsize=(6, 5))
plt.title("weights of the model")    # 模型的权重
plt.plot(clf.coef_, color="lightgreen", linewidth=lw, label="Bayesian Ridge estimate")
plt.plot(w, color="gold", linewidth=lw, label="Ground truth")
plt.plot(ols.coef_, color="navy", linestyle="--", label="OLS estimate")
plt.xlabel("Features")
plt.ylabel("Values of the weights")
plt.legend(loc="best", prop=dict(size=12))

plt.figure(figsize=(6, 5))
plt.title("Histogram of the weights")
plt.hist(clf.coef_, bins=n_features, color="gold", log=True)
plt.scatter(clf.coef_[relevant_features], 5 * np.ones(len(relevant_features)), color="navy", label="Relevant features")
plt.ylabel("Features")
plt.xlabel("Values of the weights")
plt.legend(loc="upper left")

plt.figure(figsize=(6, 5))
plt.title("Marginal log-likelihood")
plt.plot(clf.scores_, color="navy", linewidth=lw)
plt.ylabel("Score")
plt.xlabel("Iterations")

plt.show()