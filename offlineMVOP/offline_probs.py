import numpy as np
'''
本程序用于计算真实目标函数和真实约束值及其标签
搜索范围： [-5.12, 5.12]
'''

# 线性约束Ellipsoid问题
def Ellipsoid01(x):
    d = x.shape[1]
    n_d = int(0.5*d)
    Q = np.arange(-5, 5)

    # 解码
    tmp = x.copy()
    for i in range(x.shape[0]):
        for j in range(n_d):
            tmp[i, j] = Q[round(x[i, j])]

    #计算目标值
    f = np.zeros(tmp.shape[0])
    for i in range(1,d+1):
        f += i*(tmp[:,i-1]**2)

    #计算约束值与标签
    cons = np.zeros((tmp.shape[0], 2))
    label = []
    for i in range(tmp.shape[0]):
        cons[i,0] = (5 - np.sum(tmp[i, :]))/d
        cons[i,1] = np.sum(tmp[i, :int(d / 2)]) / int(d / 2) - np.sum(tmp[i, int(d / 2):]) / (d - int(d / 2))
        if (cons[i,0]  <= 0 and cons[i,1]  <= 0):
            label.append(0)
        else:
            label.append(1)
    return f, cons, np.array(label)


# 非线性约束Ellipsoid问题
def Ellipsoid02(x):
    d = x.shape[1]
    n_d = int(0.5 * d)
    Q = np.arange(-5, 5)

    # 解码
    tmp = x.copy()
    for i in range(x.shape[0]):
        for j in range(n_d):
            tmp[i, j] = Q[round(x[i, j])]

    # 计算目标值
    f = np.zeros(tmp.shape[0])
    for i in range(1,d+1):
        f += i*(tmp[:,i-1]**2)

    # 计算约束值与标签
    cons = np.zeros((tmp.shape[0], 2))
    label = []
    for i in range(tmp.shape[0]):
        cons[i,0] = 9*d - np.sum(tmp[i, :]**2)
        cons[i,1] = np.sum(tmp[i, :int(d / 2)]**2) / int(d / 2) - (np.sum(tmp[i, int(d / 2):]**2) / (d - int(d / 2)))
        if (cons[i,0]<=0 and cons[i,1]<= 0):
            label.append(0)
        else:
            label.append(1)
    return f, cons, np.array(label)


#线性约束Rastrigin问题
def Rastrigin01(x):
    d = x.shape[1]
    n_d = int(0.5 * d)
    Q = np.arange(-5, 5)

    # 解码
    tmp = x.copy()
    for i in range(x.shape[0]):
        for j in range(n_d):
            tmp[i, j] = Q[round(x[i, j])]

    # 计算目标值
    s = np.sum(tmp ** 2 - 10 * np.cos(2 * np.pi * tmp), axis=1)
    f = 10 * d + s

    # 计算约束值与标签
    cons = np.zeros((tmp.shape[0], 2))
    label = []
    for i in range(tmp.shape[0]):
        cons[i, 0] = (5 - np.sum(tmp[i, :]))/d
        cons[i, 1] = np.sum(tmp[i, :int(d / 2)]) / int(d / 2) - np.sum(tmp[i, int(d / 2):]) / (d - int(d / 2))
        if (cons[i, 0] <= 0 and cons[i, 1] <= 0):
            label.append(0)
        else:
            label.append(1)
    return f, cons, np.array(label)

#非线性约束Rastrigin问题
def Rastrigin02(x):
    d = x.shape[1]
    n_d = int(0.5 * d)
    Q = np.arange(-5, 5)

    # 解码
    tmp = x.copy()
    for i in range(x.shape[0]):
        for j in range(n_d):
            tmp[i, j] = Q[round(x[i, j])]

    # 计算目标值
    s = np.sum(tmp ** 2 - 10 * np.cos(2 * np.pi * tmp), axis=1)
    f = 10 * d + s

    # 计算约束值与标签
    cons = np.zeros((tmp.shape[0], 2))
    label = []
    for i in range(tmp.shape[0]):
        cons[i, 0] = 10 * d - np.sum((tmp[i, :] + 1) ** 2)
        cons[i, 1] = 10 * d - np.sum((tmp[i, :] - 1) ** 2)

        if (cons[i, 0] <= 0 and cons[i, 1] <= 0):
            label.append(0)
        else:
            label.append(1)

    return f,cons,np.array(label)










