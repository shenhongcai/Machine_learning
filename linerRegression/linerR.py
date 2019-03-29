"""   
@Project Name: ML
@Author: Shen Hongcai
@Time: 2019-03-28, 20:51
@Python Version: python3.6
@Coding Scheme: utf-8
@Interpreter Name: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
# 自己构造数据


_learning_rate = 0.001
_iter_number = 1000
_data_path = "../dataset/data.csv"


def data(file_path):
    file=np.loadtxt(file_path, delimiter=",")
    return file


data = data(_data_path)


def update_gradient(w, b, data, lr):
    gradient_w = 0
    gradient_b = 0
    N = len(data)
    x = data[:, 0]
    y = data[:, 1]
    for i in range(len(data)):
        gradient_w += (1/N)*x[i]*((w*x[i]+b)-y[i])
        gradient_b += (1/N)*(w*x[i]+b-y[i])
    update_w = w-lr*gradient_w
    update_b = b-lr*gradient_b
    return [update_w, update_b]


def loss(w, b, data):
    x = data[:, 0]
    y = data[:, 1]
    loss_function = ((w*x+b)-y)**2
    total_loss = np.sum(loss_function, axis=0)
    return total_loss/len(data)   # 平均损失


def fit():
    w = np.random.random()   # 随机初始化权重
    b = np.random.random()
    for i in range(_iter_number):
        w, b = update_gradient(w, b, data, _learning_rate)
        if i % 50 == 0:
            print("迭代%d次后，训练误差为：%.3f" % (i, loss(w, b, data)))
    print("[w,b]:", (w, b))
    plot_data(data, b, w)




def plot_data(data,b,k):
    #plotting
    x = data[:, 0]
    y = data[:, 1]
    y_predict = k*x+b

    plt.plot(x, y, 'o')
    plt.plot(x, y_predict, 'k-')
    plt.show()






fit()