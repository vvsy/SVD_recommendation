import numpy as np
import pandas as pd
import random


class SVDPP:
    def __init__(self, mat, K=20):
        self.mat = np.array(mat)
        self.K = K
        self.bi = {}
        self.bu = {}
        self.qi = {}
        self.pu = {}
        self.avg = np.mean(self.mat[:, 2])
        self.y = {}
        self.u_dict = {}
        for i in range(self.mat.shape[0]):

            uid = self.mat[i, 0]
            iid = self.mat[i, 1]
            self.u_dict.setdefault(uid, [])
            self.u_dict[uid].append(iid)
            self.bi.setdefault(iid, 0)
            self.bu.setdefault(uid, 0)
            self.qi.setdefault(iid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.pu.setdefault(uid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.y.setdefault(iid, np.zeros((self.K, 1)) + .1)

    def predict(self, uid, iid):
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.zeros((self.K, 1)))
        self.pu.setdefault(uid, np.zeros((self.K, 1)))
        self.y.setdefault(uid, np.zeros((self.K, 1)))
        self.u_dict.setdefault(uid, [])
        u_impl_prf, sqrt_Nu = self.getY(uid, iid)
        # 這裡將用戶bias加重1.5倍
        score = self.avg + self.bi[iid] + 1.5 * self.bu[uid] + np.sum(self.qi[iid] * (self.pu[uid] + u_impl_prf))
        if score > 10:
            score = 10
        if score < 1:
            score = 0

        return score

    def getY(self, uid, iid):
        Nu = self.u_dict[uid]
        I_Nu = len(Nu)
        sqrt_Nu = np.sqrt(I_Nu)
        y_u = np.zeros((self.K, 1))
        if I_Nu == 0:
            u_impl_prf = y_u
        else:
            for i in Nu:
                y_u += self.y[i]
            u_impl_prf = y_u / sqrt_Nu

        return u_impl_prf, sqrt_Nu

    def train(self, steps=20, gamma=0.04, Lambda=0.15):
        print('train data size', self.mat.shape)
        for step in range(steps):
            print('step', step + 1, 'is running')
            KK = np.random.permutation(self.mat.shape[0])
            rmse = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                uid = self.mat[j, 0]
                iid = self.mat[j, 1]
                score = self.mat[j, 2]
                predict = self.predict(uid, iid)
                u_impl_prf, sqrt_Nu = self.getY(uid, iid)
                eui = score - predict
                rmse += eui**2
                self.bu[uid] += gamma * (eui - Lambda * self.bu[uid])
                self.bi[iid] += gamma * (eui - Lambda * self.bi[iid])
                self.pu[uid] += gamma * (eui * self.qi[iid] - Lambda * self.pu[uid])
                self.qi[iid] += gamma * (eui * (self.pu[uid] + u_impl_prf) - Lambda * self.qi[iid])
                for j in self.u_dict[uid]:
                    self.y[j] += gamma * (eui * self.qi[j] / sqrt_Nu - Lambda * self.y[j])

            gamma = 0.93 * gamma
            print('rmse is', np.sqrt(rmse / self.mat.shape[0]))

    def test(self, test_data):

        rec_iid = []
        rec_score = []
        rec_top10 = {}
        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        rmse = 0.0

        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            score = test_data[i, 2]
            # 推薦用戶前10高購買可能物品
            rec_iid.append(iid)
            rec_score.append(score)
            rec_top10 = {"iid": rec_iid, "score": rec_score}
            rec_top10 = pd.DataFrame(rec_top10).sort_values(by='score', ascending=False)[:10]
            final_uid = [uid]
            final_uid.extend(list(rec_top10.iid))
            print(final_uid)
            eui = score - self.predict(uid, iid)
            rmse += eui**2

        print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))


def splitDATA():

    # 隱私設定檔案名稱
    filepath = ".csv"
    df = pd.read_csv(filepath)

    testid = random.choices(df.u.unique(), k=1000)

    train_df = df[~df.u.isin(testid)]
    test_df = df[df.u.isin(testid)]
    return train_df, test_df


train_df, test_df = splitDATA()


a = SVDPP(train_df, 20)
a.train()
a.test(test_df)
