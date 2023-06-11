import os
import matplotlib.pyplot as plt
from numpy import dtype
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import time as t
# from WGAN_DG import WGAN_GP
from WGAN import WGAN_GP
from clients import ClientsGroup, client
from config import parse_args

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

if __name__=="__main__":
    args = parse_args()
    t_begin = t.time()
    test_mkdir(args.save_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = WGAN_GP(args)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
        #Pytorch中的多GPU并行计算是数据级并行，相当于开了多个进程，每个进程自己独立运行，然后再整合在一起。
        #多GPU计算的前提是你的计算机上得有多个GPU,在cmd上输入nvidia-smi来查看自己的设备上的GPU信息。
        #如果多GPU运算时，程序花费的时间反而更多了，这其实是因为你的batch_size太小了，
        #因为torch.nn.DataParallel()这个函数是将每个batch的数据平均拆开分配到多个GPU上进行计算，计算完再返回来合并。
        #这导致GPU之间的开关和通讯过程占了大部分的时间开销。
    else:
        print("这里没有多线程")

    # net = net.to(dev)   #'WGAN_GP' object has no attribute 'to'
    # clients = ['西溪湿地', '双浦镇', '转塘街道','知音路']
    clients = ['杭州市', '双浦镇', '金华市', '丽水市']
    myClients = ClientsGroup(clients, 1)

    num_in_comm = 4 #选择4个客户端
    #cfraction：0 means 1 client, 1 means total clients

    Gglobal_parameters = {}
    Dglobal_parameters = {}
    for key, var in net.G.state_dict().items():
        Gglobal_parameters[key] = var.clone()

    for key, var in net.D.state_dict().items():
        Dglobal_parameters[key] = var.clone()

    for i in range(args.num_comm):  #num_comm通信次数  20
        print("communicate round {}".format(i+1))

        order = np.random.permutation(args.num_of_clients)  #对所有client进行随机排序
        clients_in_comm = [clients[i] for i in order[0:4]]
        print("clients_in_comm:",clients_in_comm)
        Gsum_parameters = None
        Dsum_parameters = None
        #每个Client基于当前模型参数和自己的数据训练并更新模型，返回每个Client更新后的参数
        for client in tqdm(clients_in_comm):
            #获取当前Client训练得到的参数
            Glocal_parameters,Dlocal_parameters = myClients.clients_train[client].localUpdate(args.epoch,
                                                                        net, Gglobal_parameters, Dglobal_parameters,client,i)
            #print("本地参数:",Glocal_parameters.items())
            #对所有的client返回的参数累加(最后取平均值)
            if Gsum_parameters is None:
                Gsum_parameters = {}
                for key, var in Glocal_parameters.items():
                    Gsum_parameters[key] = var.clone()
            else:
                for var in Gsum_parameters:
                    Gsum_parameters[var] = Gsum_parameters[var] + Glocal_parameters[var]

            if Dsum_parameters is None:
                Dsum_parameters = {}
                for key, vard in Dlocal_parameters.items():
                    Dsum_parameters[key] = vard.clone()
            else:
                for vard in Dsum_parameters:
                    Dsum_parameters[vard] = Dsum_parameters[vard] + Dlocal_parameters[vard]
        #取平均值，得到本次通信中Server得到的更新后的模型参数
        for var in Gglobal_parameters:
            Gglobal_parameters[var] = (Gsum_parameters[var] / num_in_comm)
        # 取平均值，得到本次通信中Server得到的更新后的模型参数
        for vard in Dglobal_parameters:
            Dglobal_parameters[vard] = (Dsum_parameters[vard] / num_in_comm)
    t_end = t.time()
    print('Time of training-{}'.format((t_end - t_begin)))

    print('模型训练完毕,开始测试...')
    net.G.load_state_dict(Gglobal_parameters, strict=True)
    net.D.load_state_dict(Dglobal_parameters, strict=True)

    # 异常识别实验测试部分

    # MAE_Loss_list = []
    # RMSE_Loss_list = []
    # MAPE_Loss_list = []
    # MSE_Loss_list = []
    # for client in myClients.clientnames:
    #     test_dl = DataLoader(myClients.clients_detection[client].dataloader, batch_size=1, shuffle=True, num_workers=0)
    #     for m, batch in enumerate(test_dl):
    #         net.D.zero_grad()
    #         hydrology , label = batch
    #         hydrology = hydrology.type(torch.FloatTensor)
    #         hydrology, label = net.get_torch_variable(hydrology), net.get_torch_variable(label)
    #         pred = net.D(hydrology)
    #         # print(client,f' pred: {pred}, label:{label}')
    #
    #         pred = pred.sum()
    #         label = label.sum()
    #
    #         MSE_loss = net.MSE(pred.float(),label.float())
    #
    #         f = open('newdec.txt', 'a')
    #         mseloss = MSE_loss.item()
    #         f.write(str(mseloss))
    #         f.write('\n')
    #         f.close()
    #
    #         MSE_loss = MSE_loss.requires_grad_()
    #
    #         MAE_loss = net.mae_value(pred, label)
    #         RMSE_loss = net.rmse_value(pred, label)
    #         MAPE_loss = net.mape_value(pred, label)
    #
    #         MAE_Loss_list.append(MAE_loss)
    #         RMSE_Loss_list.append(RMSE_loss)
    #         MAPE_Loss_list.append(MAPE_loss)
    #         MSE_Loss_list.append(MSE_loss)
    #
    #         MSE_loss.backward()
    #         net.d_optimizer.step()
    #         print(client,":MAEloss:%f    RMSEloss:%f    MAPEloss:%f" % (MAE_loss,RMSE_loss,MAPE_loss))
    #         f = open('detection.txt', 'a')
    #         f.write('\nclient:%s    pred:%f    label:%f' % (client,pred,label))
    #         f.close()
    #
    # MAE_loss_Avg = sum(MAE_Loss_list) / len(MAE_Loss_list)
    # RMSE_loss_Avg = sum(RMSE_Loss_list) / len(RMSE_Loss_list)
    # MSE_loss_Avg = sum(MSE_Loss_list) / len(MSE_Loss_list)
    # MAPE_loss_Avg = sum(MAPE_Loss_list) / len(MAPE_Loss_list)
    # f = open('detection.txt', 'a')
    # f.write('\nMAEloss_Avg:%f    MSEloss_Avg:%f    RMSEloss_Avg:%f    MAPEloss_Avg:%f' % (MAE_loss_Avg, MSE_loss_Avg, RMSE_loss_Avg, MAPE_loss_Avg))
    # f.close()

    # fig = plt.figure(figsize=(12, 6), dpi=100)
    # plt.subplot(2, 2, 1)
    # plt.plot(MSE_Loss_list, color='firebrick', linewidth=0.8);
    # plt.legend(loc='upper center')
    # plt.ylabel('Loss', fontsize=8);
    # plt.xlabel('Numbers')
    # plt.title('MSE_Loss')
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(MAPE_Loss_list, color='blue', linewidth=0.8);
    # plt.legend(loc='upper center')
    # plt.title('MAPE_Loss');
    # plt.ylabel('Loss', fontsize=8);
    # plt.xlabel('Numbers')
    # plt.subplot(2, 1, 2)
    # plt.plot(MAE_Loss_list, '--.', c='green',linewidth=0.8)
    # plt.ylabel('Loss', fontsize=8)
    # plt.title('MAE_Loss');
    # plt.xlabel('Numbers')
    # plt.axhspan(0, 0.1)

    # adjust the space between subplots...(w:width, h:height)
    # plt.subplots_adjust(hspace=0.6, wspace=0.4)
    # plt.grid(True, linestyle='--', linewidth=1, alpha=0.5)
    #
    # # add a title to the whole figure
    # fig.suptitle('Anomaly Detection Performance', fontsize=16)
    # plt.savefig('hy_Detection.png')

    #数据修复实验测试部分
    MAELoss_list = []
    RMSELoss_list = []
    MAPELoss_list = []
    for client in myClients.clientnames:
        test_dl = DataLoader(myClients.clients_repair[client].dataloader, batch_size=1, shuffle=True, num_workers=0)
        # sum_loss = 0
        for m, hydrology in enumerate(test_dl):
            hydrology = hydrology.type(torch.FloatTensor)
            z = torch.rand((1, 3, 1))
            hydrology, z = net.get_torch_variable(hydrology), net.get_torch_variable(z)
            fake_hydrology = net.G(z)
            fake = torch.sum(fake_hydrology)/len(fake_hydrology)
            true_hy = torch.sum(hydrology)/len(hydrology)

            MAEloss = net.mae_value(fake_hydrology, hydrology)
            f = open('MAE.txt', 'a')
            maeloss = MAEloss.item()
            f.write(str(maeloss))
            f.write('\n')
            f.close()

            RMSEloss = net.rmse_value(fake_hydrology, hydrology)
            f = open('RMSE.txt', 'a')
            rmseloss = RMSEloss
            f.write(str(rmseloss))
            f.write('\n')
            f.close()

            MAPEloss = net.remape_value(fake, true_hy)
            f = open('MAPE.txt', 'a')
            mapeloss = MAPEloss.item()
            f.write(str(mapeloss))
            f.write('\n')
            f.close()

            MAELoss_list.append(MAEloss)
            RMSELoss_list.append(RMSEloss)
            MAPELoss_list.append(MAPEloss)
            print(client,":MAEloss:%f    RMSEloss:%f    MAPEloss:%f" % (MAEloss,RMSEloss,MAPEloss))
            f = open('repair.txt', 'a')
            f.write('\n客户端%s第%d条测试数据    MAEloss:%f    RMSEloss:%f    MAPEloss:%f' % (client,m,MAEloss,RMSEloss,MAPEloss))
            f.close()
    # print('Time of training-{}'.format((t_end - t_begin)))
    # print('Time of training-{}'.format((t_end - t_begin)))
    # print('Time of training-{}'.format((t_end - t_begin)))
    # print('Time of training-{}'.format((t_end - t_begin)))

    MAEloss_Avg = sum(MAELoss_list) / len(MAELoss_list)
    RMSEloss_Avg = sum(RMSELoss_list) / len(RMSELoss_list)
    MAPEloss_Avg = sum(MAPELoss_list) / len(MAPELoss_list)
    f = open('repair.txt', 'a')
    f.write('\nMAEloss_Avg:%f    RMSEloss_Avg:%f    MAPEloss_Avg:%f' % (MAEloss_Avg, RMSEloss_Avg, MAPEloss_Avg))
    f.close()

    fig = plt.figure(figsize=(12, 6), dpi=100)


    # plt.rc('font', family='Times New Roman')
    # ax = plt.axes()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    plt.subplot(2, 2, 1)
    plt.plot(RMSELoss_list, color='firebrick', linewidth=1.0)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.ylabel('RMSE_Loss', fontsize=10)
    plt.xlabel('Numbers')

    plt.subplot(2, 2, 2)
    plt.plot(MAPELoss_list, color='blue', linewidth=1.0)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.ylabel('MAPE_Loss', fontsize=10)
    plt.xlabel('Numbers')

    plt.subplot(2, 1, 2)
    plt.plot(MAELoss_list, '--.', c='green',linewidth=1.0)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.ylabel('MAE_Loss', fontsize=10)
    plt.xlabel('Numbers')
    # plt.axhspan(0, 0.1)

    # adjust the space between subplots...(w:width, h:height)
    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    # plt.grid(True, linestyle='--', linewidth=1, alpha=0.5)

    # add a title to the whole figure
    # fig.suptitle('Data Repair Performance', fontsize=16)

    plt.savefig('hy_Repair.png')
    # plt.show()













