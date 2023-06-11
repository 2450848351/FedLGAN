import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import hydrology_Dataset


class client(object):
    def __init__(self, dataloader, dev):
        self.dataloader = dataloader
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None


    def localUpdate(self, localEpoch, Net, Gglobal_parameters , Dglobal_parameters , client , servertimes):
        # :param
        # localEpoch: 当前Client的迭代次数
        # client=[client1,client2,client3,client4] 通过字符串划分数据集
        # :param
        # Net: Server共享的模型
        # :param
        # lossFun: 损失函数
        # :param
        # opti: 优化函数
        # :param
        # Gglobal_parameters: 当前通信中最新全局参数
        # Dglobal_parameters: 当前通信中最新全局参数

        # 加载当前通信中最新全局参数
        Net.G.load_state_dict(Gglobal_parameters, strict=True)
        Net.D.load_state_dict(Dglobal_parameters, strict=True)

        self.train_dl = DataLoader(self.dataloader, batch_size=64, shuffle=True, num_workers=0)
        # 设置迭代次数
        for epoch in range(localEpoch):
                data = Net.get_infinite_batches(self.train_dl)
                one = torch.tensor(1, dtype=torch.float)
                #torch.FloatTensot()默认生成32位浮点数，dtype=torch.float32或者torch.float
                mone = one * -1
                for p in Net.D.parameters():
                    p.requires_grad = True

                d_loss_real = 0
                d_loss_fake = 0
                Wasserstein_D = 0
                print("%s   epoch:%d" %(client,epoch))
                for d_iter in range(20):
                    Net.D.zero_grad()
                    hydrology = data.__next__()  # 64*3*3
                    # hydrology = np.expand_dims(hydrology, axis=1)  # 64*1*3*3
                    # hydrology = torch.from_numpy(hydrology)  # 这里的hydrology是DoubleTensor
                    hydrology = hydrology.type(torch.FloatTensor)

                    if (hydrology.size()[0] != Net.batch_size):  # Check for batch to have full batch_size
                        continue

                    z = torch.rand((Net.batch_size, 3, 1))

                    hydrology, z = Net.get_torch_variable(hydrology), Net.get_torch_variable(z)

                    # Train discriminator
                    # WGAN - Training discriminator more iterations than generator
                    # Train with real hydrology
                    d_loss_real = Net.D(hydrology)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    # Train with fake hydrology
                    z = Net.get_torch_variable(torch.randn(Net.batch_size, 3, 1))

                    fake_hydrology = Net.G(z)
                    d_loss_fake = Net.D(fake_hydrology)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)

                    # Train with gradient penalty
                    # if d_iter%10==0:
                    #     f = open('traindata.txt', 'a')
                    #     f.write('\n判别器_fake_hydrology:')
                    #     f.write('\n%s' % fake_hydrology[0])
                    #     f.write('\n判别器_true_hydrology:')
                    #     f.write('\n%s' % hydrology[0])
                    #     f.close()

                    gradient_penalty = Net.calculate_gradient_penalty(hydrology.data, fake_hydrology.data)
                    # gradient_penalty.backward() 不支持双RNN反向传播，因为此时生成器和判别器都是lstm组成

                    MAEloss = Net.mae_value(fake_hydrology, hydrology)


                    # d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    # Wasserstein_D = d_loss_real - d_loss_fake
                    Net.d_optimizer.step()
                    print(
                        f'Discriminator iteration: {d_iter}/{20}, Servertimes:{servertimes}, MAEloss:{MAEloss}, d_loss_fake:{d_loss_fake}, d_loss_real: {d_loss_real}')
                for p in Net.D.parameters():
                    p.requires_grad = False  # to avoid computation

                Net.G.zero_grad()
                # train generator
                # compute loss with fake hydrology
                z = Net.get_torch_variable(torch.randn(Net.batch_size, 3, 1))
                fake_hydrology = Net.G(z)

                g_loss = Net.D(fake_hydrology)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                # g_cost = -g_loss
                Net.g_optimizer.step()
                print(f'Generator iteration: {epoch}/{localEpoch}, g_loss: {g_loss}')
        return Net.G.state_dict(),Net.D.state_dict()
        #.state_dict()是一个字典对象，将每一层与它的对应参数建立映射关系，
        #优化器对象opti也有一个state_dict，它包含了优化器的状态以及被使用的超参数

    def local_val(self):
        pass

class ClientsGroup(object):
    def __init__(self, clientnames, dev):  #clientnames=[“client1”,'client2','client3','client4']
        self.clientnames = clientnames
        self.dev = dev
        self.clients_train = {}
        self.clients_repair = {}
        self.clients_detection = {}
        # self.address_sid = {'西溪湿地': '1407', '双浦镇': '3488', '五里塘路': '16295','知音路': '9709'}
        self.address_sid = {'杭州市': '1407', '双浦镇': '3488', '金华市': '2175', '丽水市': '1486'}

        self.trainAllocation()
        self.repairtAllocation()
        self.detectionAllocation()

    def repairtAllocation(self):  #数据修复分配测试集
        for i,clientname in enumerate(self.clientnames):
            testdataloader = hydrology_Dataset(sid=self.address_sid[clientname], model='test',methods='repair')
            someone = client(testdataloader, self.dev)
            self.clients_repair[clientname] = someone

    def detectionAllocation(self):  #异常检测分配测试集
        for i,clientname in enumerate(self.clientnames):
            testdataloader = hydrology_Dataset(sid=self.address_sid[clientname], model='test',methods='detection')
            someone = client(testdataloader, self.dev)
            self.clients_detection[clientname] = someone

    def trainAllocation(self):  #分配训练集
        for i,clientname in enumerate(self.clientnames):
            traindataloader = hydrology_Dataset(sid=self.address_sid[clientname], model='train',methods='repair')
            someone = client(traindataloader, self.dev)
            self.clients_train[clientname] = someone

if __name__=="__main__":  #1407 3488 3489 7642  9709
    # address_sid = {'西溪湿地': 1407, '双浦镇': 3488, '转塘街道': 7642,'知音路': 9709}
    # clients = ['西溪湿地','双浦镇','转塘街道','知音路']
    # print(address_sid[clients[0]])   #根据键获取值
    # c= list(address_sid.keys())[list(address_sid.values()).index(1407)]  #根据值获取键
    # print(c)
    # clients = ['西溪湿地', '双浦镇', '转塘街道', '知音路']
    clients = ['西溪湿地', '双浦镇', '五里塘路', '知音路']
    MyClients = ClientsGroup(clients, 1)
    traindataloader = DataLoader(MyClients.clients_detection['知音路'].dataloader, batch_size=1, shuffle=True,
                            num_workers=0)
    for i, batch in enumerate(traindataloader):
        array , label = batch
        print("i：",batch)
        print(label)







