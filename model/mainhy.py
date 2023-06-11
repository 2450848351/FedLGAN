from torch.utils.data import DataLoader
from datasets import hydrology_Dataset

from config import parse_args
from WGAN_DG import WGAN_GP

def main(args):
    model = WGAN_GP(args)
    traindataloader = DataLoader(hydrology_Dataset('1407', model="train"), batch_size=64, shuffle=True,
                                 num_workers=0)
    # shuffle控制是否随机index
    testdataloader = DataLoader(hydrology_Dataset('1407', model="test"), batch_size=64, shuffle=True,
                                 num_workers=0)
    if args.is_train == 'True':   # 根据传入的参数判断是否为模型训练阶段，训练阶段：训练模型参数；评估阶段：根据输入数据、模型参数得到结果
        model.train(traindataloader,testdataloader)
    # start evaluating on test data
    else:
        model.evaluate(testdataloader, args.load_D, args.load_G)

    # for i, batch in enumerate(testdataloader):
    #     Adj, fea, label = batch
    #     print(i)
    #     print(Adj)
    #     # print(torch.mean(Adj.float()))
    #     print(fea)
    #     print(label)
if __name__ == '__main__':
    args = parse_args()
    main(args)



