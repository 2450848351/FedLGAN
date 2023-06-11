import pickle
import time
import pandas as pd
# import pymysql
import numpy as np

import torch

class CreateCsv:
    def __init__(self, filename, sid):
        self.filename = filename
        self.sid = sid
        self.csvpath = "../data/hy_" + self.sid + "/hy_" + self.sid + ".csv"
        self.pklpath = "../data/hy_" + self.sid + "/hy_" + self.sid + ".pkl"
        self.pkltestpath = "../data/hy_" + self.sid + "/hytest_" + self.sid + ".pkl"


    def splited(self):
        input = self.filename.split('_')[1]
        output = ''.join(filter(str.isdigit, input))
        return output

    def process_1(self):  #将数据库中的数据读出来并放进CSV文件
        conn = pymysql.connect(host="localhost", port=3306, user='root', password='123', db="hydrology", charset="utf8")
        sql = "select dt,group_concat(item,'_',val order BY `index` desc SEPARATOR ',') items from " + self.filename + " group by dt;"
        # sql = "select dt,group_concat(val order BY `index` desc SEPARATOR ',') items from " + self.filename + " group by dt;"
        df = pd.read_sql(sql, conn)
        # 转化为csv
        df.to_csv(path_or_buf=self.csvpath, float_format=8, header=False,index=False, sep='/',mode='a')
        #mode='a'，不覆盖原有内容写入
        print("Finish")

    # def process_2(self):    #将CSV文件中的格式转换
    #     f = pd.read_csv(self.csvpath,names=['a', 'b'], sep='/', index_col='a', encoding='utf8')
    #     for index, row in f.iterrows():
    #         numberarray = np.zeros((3, 3))
    #         # datetime = strtransformdate(index)
    #         numberarray[0][0]=float(self.sid)/100    #防止张量中数据过大
    #         numberarray[0][1]=datesplited(index)   #由于原来的datetime时间戳会导致数据过大，直接存入严重影响矩阵，先考虑采用
    #         # print('datetime:',datetime)
    #         strlist =row.str.split(',')
    #         for i, n in enumerate(strlist[0]):
    #             createarray(n, numberarray)
    #         numbertensor = torch.tensor(numberarray)
    #         pickle.dump(numbertensor, self.pklopen)
    #         print("数组d:",numberarray)

    def process_2(self):    #将CSV文件中的格式转换
        pklopen = open(self.pklpath, 'wb')
        f = pd.read_csv(self.csvpath,names=['a', 'b'], sep='/', index_col='a', encoding='utf8')
        for index, row in f.iterrows():
            if(judge(row)==1):
                numberarray = np.zeros((3, 3))
            # datetime = strtransformdate(index)
                numberarray[0][0]=float(self.sid)/100    #防止张量中数据过大
                numberarray[0][1]=datesplited(index)   #由于原来的datetime时间戳会导致数据过大，直接存入严重影响矩阵，先考虑采用
            # print('datetime:',datetime)
                strlist =row.str.split(',')
                for i, n in enumerate(strlist[0]):
                    createarray(n, numberarray)
                newnumberarray = noramlization(numberarray)
                numbertensor = torch.tensor(newnumberarray)
                print("张量:", numbertensor)
                pickle.dump(numbertensor, pklopen)

    def process_3(self):    #将CSV文件中的格式转换 提取测试集
        pkltestopen = open(self.pkltestpath, 'wb')
        f = pd.read_csv(self.csvpath,names=['a', 'b'], sep='/', index_col='a', encoding='utf8')
        for index, row in f.iterrows():
            label = 0
            if(judge(row)==1):
                label = 1
            numberarray = np.zeros((3, 3))
            # numberarray = np.ones((3, 3))
            modify(numberarray) #把矩阵中的全部元素置为-1
            # datetime = strtransformdate(index)
            numberarray[0][0]=float(self.sid)/100    #防止张量中数据过大
            numberarray[0][1]=datesplited(index)   #由于原来的datetime时间戳会导致数据过大，直接存入严重影响矩阵
            numberarray[0][2] = 0.01 #防止某一列全为相同的数导致出现nan
            # print('datetime:',datetime)
            strlist =row.str.split(',')
            for i, n in enumerate(strlist[0]):
                createarray(n, numberarray)
            newnumberarray = noramlization(numberarray)
            numbertensor = torch.tensor(newnumberarray)
            tuple = (numbertensor,label)
            print("张量:", tuple)
            pickle.dump(tuple, pkltestopen)

def modify(array):
    for i in range(0, 3):
        for j in range(0, 3):
            array[i][j] = -1

def judge(row):
    flag = 0
    for i, v in row.items():
        if v.find('CSQ') != -1 or v.find('当前水位') != -1 or v.find('雨量计数') != -1 or v.find('环境温度') != -1:
            flag = 1
    return flag

def spliter(inputstr):
    input = inputstr.split('_')[1]
    output = ''.join(filter(str.isdigit, input))
    return output

def noramlization(data):   #归一化处理
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    # return normData, ranges, minVals
    return normData

def createarray(str,array):
    strlist = str.split('_')
    # flag = np.array([['日期','日水位','时水位','水位5分钟'],
    #              ['当前水位','日雨量','时雨量','雨量5分钟'],
    #              ['雨量计数','CSQ','环境温度','电压']])
    # 原来的根据监测标签张量值的存入  构造的是3*4的矩阵，由于雨量5分钟，时雨量，日雨量值/水位5分钟，时水位，日水位总是一致，先考虑去除
    flag = np.array([['SID','日期','CSQ'],
                 ['当前水位','水位5分钟','电压'],
                 ['雨量计数','雨量5分钟','环境温度']])
    address = np.where(flag == strlist[0])   #根据strlist[0]确定在strlist[1]的值在array中的位置
    if np.any(address):
        a,b = address[0][0],address[1][0]
        array[a][b] = np.float64(strlist[1])

    # return a

def datesplited(date):  #只记录月日,比如2022-03-10 23:50:00->3.10
    out = date.split(' ')
    output = out[0].split('-')
    a = int(output[1])
    b = int(output[2])
    return float(a)+float(b/100)

def datesplit(date):   #将datetime格式的日期转换为数字：2022-03-10 23:50:00 -> 20220310235000
    output = ''.join(filter(str.isdigit, date))
    return output

def strtransformdate(str):
    input = time.strptime(str,'%Y-%m-%d %H:%M:%S')#str转换为datetime
    output = int(time.mktime(input))#datetime转换为时间戳
    return output

if __name__ == '__main__':
     # c = CreateCsv('t_real20220309', '1407')
     #
     # c.process_2()
     # print('结束了')
    a = spliter('t_real20220309')
    for i in range(14):
        tablenumber = int(a)+i
        tablename = 't_real'+str(tablenumber)
        c = CreateCsv(tablename, '1486')
        # c.process_1()
        c.process_2()
        c.process_3()
    print('结束了!!!')

