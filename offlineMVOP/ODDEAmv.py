import csv
import random
import numpy as np
from offline_probs import Ellipsoid01, Ellipsoid02, Rastrigin01, Rastrigin02
from SMs.RBFN import RBFN

from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


class Pop:
    def __init__(self, X, objv=None, cv=None, label=None):
        self.X = X
        self.objv = objv
        self.cv = cv
        self.label = label


class ODDEAmv(object):
    def __init__(self, max_iter, popsize, dim, func, model_name, prob_name, rs):

        self.max_iter = max_iter
        self.popsize = popsize
        self.func = func
        self.dim = dim
        self.n_d = int(0.5 * self.dim)
        self.n_c = self.dim - self.n_d

        self.clb = np.append(np.zeros(self.n_d), -5.12 * np.ones(self.n_c))
        self.cub = np.append((10 - 1) * np.ones(self.n_d), 5.12 * np.ones(self.n_c))

        self.proC = 1
        self.disC = 20
        self.proM = 1
        self.disM = 20

        self.N_lst = [10 for _ in range(self.n_d)]
        self.model_name = model_name
        self.prob_name = prob_name
        self.rs = rs
        self.offline_data = []

    def get_data(self):
        size = 11*self.dim
        X = np.zeros(( size , self.dim))
        area = self.cub - self.clb

        np.random.seed(self.rs)
        for j in range(self.n_d):
            for i in range( size ):
                X[i, j] = int(np.random.uniform(i / size * self.N_lst[j],
                                                (i + 1) / size * self.N_lst[j]))
            np.random.shuffle(X[:, j])

        for j in range(self.n_d, self.dim):
            for i in range(size):
                X[i, j] = self.clb[j] + np.random.uniform(i / size * area[j],
                                                          (i + 1) / size * area[j])
            np.random.shuffle(X[:, j])
        np.random.seed()

        self.offline_data = [X, *self.func(X)]

    def initialization(self):
        X = np.zeros((self.popsize, self.dim))
        area = self.cub - self.clb

        np.random.seed(self.rs)
        for j in range(self.n_d):
            for i in range(self.popsize):
                X[i, j] = int(np.random.uniform(i / self.popsize * self.N_lst[j],
                                                (i + 1) / self.popsize * self.N_lst[j]))
            np.random.shuffle(X[:, j])

        for j in range(self.n_d, self.dim):
            for i in range(self.popsize):
                X[i, j] = self.clb[j] + np.random.uniform(i / self.popsize * area[j],
                                                          (i + 1) / self.popsize * area[j])
            np.random.shuffle(X[:, j])

        self.arch = Pop(X)

        # 初始种群评估
        self.arch.objv = self.obj_sm.predict(self.arch.X)
        self.arch.label, self.arch.cv = self.cv_predict(self.arch.X)
        self.save_data()


    def DE(self, X):
        size = X.shape[0]
        muX = np.empty((size, self.dim))

        for i in range(size):  # DE/rand/1
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = np.random.randint(0, size - 1)
                r2 = np.random.randint(0, size - 1)
                r3 = np.random.randint(0, size - 1)

            mutation = X[i] + 0.2 * (X[np.argmin(self.arch.objv)] - X[i]) + 0.5 * (X[r1] - X[r2])

            for j in range(self.dim):
                #  判断变异后的值是否满足边界条件，不满足需重新生成
                if self.clb[j] <= mutation[j] <= self.cub[j]:
                    muX[i, j] = mutation[j]
                else:
                    rand_value = self.clb[j] + np.random.random() * (self.cub[j] - self.clb[j])
                    muX[i, j] = rand_value

        crossX = np.empty((size, self.dim))
        for i in range(self.popsize):
            rj = np.random.randint(0, self.dim - 1)
            for j in range(self.dim):
                rf = np.random.random()
                if rf <= 0.8 or rj == j:
                    crossX[i, j] = muX[i, j]
                else:
                    crossX[i, j] = X[i, j]
        return crossX


    def build_surrogate(self):
        # 目标函数代理模型
        self.obj_sm = RBFN()
        self.obj_sm.fit(self.offline_data[0], self.offline_data[1])

        if self.model_name == "lr":
            self.cv_sm = [LinearRegression(), LinearRegression()]
        elif self.model_name == "svr":
            # svr 支持向量回归器(网格搜索调参)
            parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100]}
            gs1 = GridSearchCV(SVR(), parameters, cv=5,
                               scoring='r2')
            gs2 = GridSearchCV(SVR(), parameters, cv=5,
                               scoring='r2')
            gs1.fit(self.offline_data[0], self.offline_data[2][:,0])
            gs2.fit(self.offline_data[0], self.offline_data[2][:,1])
            self.cv_sm = [gs1, gs2]

        elif self.model_name == "rbfn":
            self.cv_sm = [RBFN(), RBFN()]

        elif self.model_name == "gnb":
            self.cv_sm = GaussianNB()

        elif self.model_name == "svc":
            self.cv_sm = SVC(kernel='rbf', C=100, probability=True)


        elif self.model_name == "ensemble":
            self.cv_sm = BalancedBaggingClassifier(base_estimator=SVC(kernel='rbf', C=100, probability=True),
                                                   n_estimators=10)

        # 约束违反代理模型
        if self.model_name in ['lr', 'rbfn']:
            for j, model in enumerate(self.cv_sm):
                model.fit(self.offline_data[0], self.offline_data[2][:, j])
        elif self.model_name in ['gnb', 'svc', 'ensemble']:
            self.cv_sm.fit(self.offline_data[0], self.offline_data[3])

    def cv_predict(self, X):

        if self.model_name in ['lr', 'svr', 'rbfn']:

            con1 = self.cv_sm[0].predict(X)
            con2 = self.cv_sm[1].predict(X)

            label = np.logical_and(con1 <= 0, con2 <= 0).astype(int)
            label = 1 - label

            con1 = np.maximum(con1, 0) / np.max(con1)
            con2 = np.maximum(con2, 0) / np.max(con2)
            cv = con1 + con2

        else:
            con = self.cv_sm.predict_proba(self.arch.X)
            cv = con[:, 1]
            label = (cv > 0.6).astype(int) # 0.15

        return label, cv

    def hybridCHT(self, objv, label, cv):
        # 先筛选出预估可行解
        feasible = np.where(label == 0)[0]
        nofeasible = np.array(list(set(range(len(objv))).difference(set(feasible))))

        # 预测可行个数大于种群个数
        if len(feasible) >= self.popsize:
            if (np.random.rand() < 0.445):
                indexs = feasible[np.argsort(objv[feasible])[:self.popsize]]
            else:
                indexs = feasible[random.sample(range(len(feasible)), self.popsize)] #np.argsort(cv[feasible])[:self.popsize]

        else:
            # 先筛选约束违反低的
            last = nofeasible[np.argsort(cv[nofeasible])][:self.popsize - len(feasible)]
            indexs = np.append(feasible, last)

        return indexs.tolist()

    def save_data(self):
        # 保存结果（每一代）：x_best, y_pred, cv_pred, label_pred, y_true, cv_true, label_true,
        #  y_pred_mean, y_pred_std, cv_pred_mean, cv_pred_std, y_true_mean, y_true_std, cv_true_mean, cv_true_std,

        # 最优
        feasible = np.where(self.arch.label == 0)[0]
        if len(feasible) == 0:
            ind = np.argmin(self.arch.cv)
        else:
            ind = feasible[np.argmin(self.arch.objv[feasible])]


        x_best = self.arch.X[ind].reshape(1, -1)
        y_pred = self.arch.objv[ind]
        cv_pred = self.arch.cv[ind]
        label_pred = self.arch.label[ind]

        # 平均
        y_pred_mean = np.mean(self.arch.objv)
        y_pred_std = np.std(self.arch.objv)

        cv_pred_mean = np.mean(self.arch.cv)
        cv_pred_std = np.std(self.arch.cv)

        y_true_pop, cv_true_pop, label_true_pop = self.func(self.arch.X)
        y_true_mean = np.mean(y_true_pop)
        y_true_std = np.std(y_true_pop)
        cv = np.maximum(cv_true_pop[:, 0], 0) / np.max(cv_true_pop[:, 0]) + np.maximum(cv_true_pop[:, 1],
                                                                                       0) / np.max(cv_true_pop[:, 1])
        y_true, cv_true, label_true = y_true_pop[ind], cv[ind], label_true_pop[ind]

        cv_true_mean = np.mean(cv)
        cv_true_std = np.std(cv)


        item = [x_best, y_pred, cv_pred, label_pred, y_true, cv_true, label_true, y_pred_mean, y_pred_std, cv_pred_mean,
                cv_pred_std, y_true_mean, y_true_std, cv_true_mean, cv_true_std]

        # 保存数据
        with open(self.filename, "a+", newline='', encoding='UTF-8-sig') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(item)

    def run(self):

        self.filename = f'./results/{self.prob_name}_{self.model_name}_dim={str(self.dim)}_{str(self.rs)}_T=0.6.csv'
        header = ["x_best", "y_pred", "cv_pred", "label_pred", "y_true", "cv_true", "label_true",
                  "y_pred_mean", "y_pred_std", "cv_pred_mean", "cv_pred_std", "y_true_mean", "y_true_std",
                  "cv_true_mean", "cv_true_std"]
        with open(self.filename, "w", newline='', encoding='UTF-8-sig') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(header)

        # 获取离线数据
        self.get_data()

        # 训练代理模型
        self.build_surrogate()

        # 种群初始化
        self.initialization()

        for t in tqdm(range(self.max_iter)):
            self.iter = t
            # 交叉
            # oX = self.RCGA(self.arch.X)
            oX = self.DE(self.arch.X)
            oX[:, :self.n_d] = np.round(oX[:, :self.n_d])

            # 目标和约束违反预测
            objv = self.obj_sm.predict(oX)
            label, cv = self.cv_predict(oX)

            # 合并
            merged_X = np.vstack([self.arch.X, oX])
            merged_objv = np.append(self.arch.objv, objv)
            merged_label = np.append(self.arch.label, label)
            merged_cv = np.append(self.arch.cv, cv)

            # 约束处理
            Next = self.hybridCHT(merged_objv, merged_label, merged_cv)

            # 更新种群
            self.arch.X = merged_X[Next]
            self.arch.objv = merged_objv[Next]
            self.arch.cv = merged_cv[Next]
            self.arch.label = merged_label[Next]

            # 结果保存
            self.save_data()



def run_experiment(args):
    k, func, model, prob = args
    alg = ODDEAmv(max_iter=100, popsize=100, dim=30, func=func, model_name=model, prob_name=prob, rs=k)
    alg.run()


if __name__ == "__main__":


    probs = [Ellipsoid01, Ellipsoid02,  Rastrigin01, Rastrigin02]  #
    pnames = ["Ellipsoid01", "Ellipsoid02",  "Rastrigin01",  "Rastrigin02"] #
    cv_models = ["lr", "svr", "rbfn",  "gnb", "svc", "ensemble"]  #


    n_runs = 10

    from multiprocessing.pool import Pool

    for i, func in enumerate(probs):
        pname = pnames[i]
        for model in cv_models:
            # 独立运行30次
            with Pool(n_runs) as pool:
                pool.map(run_experiment, [(i, func, model, pname) for i in range(10)])







