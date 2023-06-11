import torch
import numpy as np


def mae_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mae -- MAE 评价指标
    """
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae

a = torch.randn(2,2)
b = torch.randn(2,2)
print('a:',a)
print('b:',b)

c = mae_value(a,b)
print('mae:',c)


