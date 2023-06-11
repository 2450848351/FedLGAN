from torch.utils.data import Dataset
import pickle
import torch
import numpy as np


class hydrology_Dataset(Dataset):
    def __init__(self, sid, model, methods):
        # 改成pkl文件的路径
        self.sid = sid
        # self.pklpath = "../data/hy_" + sid + "/hytest_" + sid + ".pkl"
        self.model = model
        self.methods = methods

    # index 表示训练集的第几个
    def __getitem__(self, index):
        # print("index:",index)
        # 读取pkl文件
        if self.methods == 'detection':
            pklpath = "../data/hy_" + self.sid + "/hytest_" + self.sid + ".pkl"
        elif self.methods == 'repair':
            pklpath = "../data/hy_" + self.sid + "/hy_" + self.sid + ".pkl"
        self.file = open(pklpath, 'rb')
        # ind 总样本集中的第几个
        if self.model == 'train':
            ind = int(index / 8) * 10 + (index % 8)
        elif self.model == 'test':
            ind = int(index / 2) * 10 + (index % 2) + 8
        ind += 1
        # print('ind:',ind)
        for i in range(int(ind)):
            tuple = pickle.load(self.file)
        #返回一个样本
        return tuple

    def __len__(self):
        cnt = 1000#控制数据集中数据的条数
        if self.model == 'train':
            cnt = cnt * 0.8
        # elif self.model == 'val':
        #     cnt = cnt * 0.1
        elif self.model == 'test':
            cnt = cnt * 0.05
        return int(cnt)

def dayindetection(dataloader):
    # print("--------------------------------------------------------------")
    # print(var_name(traindataloader))
    for i, batch in enumerate(dataloader):
        array , label = batch
        print("i=",i)
        print("矩阵:",array)
        print('label:',label)

def dayinrepair(dataloader):
    # print("--------------------------------------------------------------")
    # print(var_name(traindataloader))
    for i, array in enumerate(dataloader):
        print("i=",i)
        print("矩阵:",array)

def var_name(var,all_var=locals()):#输出变量名
    return [var_name for var_name in all_var if all_var[var_name] is var][0]

if __name__=='__main__':
    from torch.utils.data import DataLoader
    testrdataloader = DataLoader(hydrology_Dataset('1407', model="test", methods='repair'), batch_size=1, shuffle=True,
                            num_workers=0)
    testddataloader = DataLoader(hydrology_Dataset('1407', model="test", methods='detection'), batch_size=1, shuffle=True,
                                 num_workers=0)
    #shuffle控制是否随机index
    # testdataloader = DataLoader(GPS_Dataset(carid='../datasets/浙AAT295', model="test"), batch_size=1, shuffle=True,
    #                         num_workers=0)

    # 迭代，遍历
    # i 就是 Adj, fea, label
    print("输出结果")
    dayinrepair(testrdataloader)
