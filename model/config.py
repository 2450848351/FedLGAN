import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

    # parser.add_argument('--model', type=str, default='WGAN-GP', choices=['GAN', 'DCGAN', 'WGAN-CP', 'WGAN-GP'])
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=4, help='numer of the clients')
    parser.add_argument('-cf', '--cfraction', type=float, default=0.1,
                        help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
    # parser.add_argument('-E', '--epoch', type=int, default=25, help='local train epoch')
    parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')
    parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
    parser.add_argument('-sf', '--save_freq', type=int, default=20,
                        help='global model save frequency(of communication)')
    parser.add_argument('-ncomm', '--num_comm', type=int, default=10, help='number of communications')
    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
    parser.add_argument('--is_train', type=str, default='True')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'fashion-mnist', 'cifar', 'stl10'],
                            help='The name of dataset')
    parser.add_argument('--download', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=5, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--cuda',  type=str, default='True', help='Availability of cuda')

    parser.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
    parser.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')
    parser.add_argument('--generator_iters', type=int, default=200, help='The number of iterations for generator in WGAN model.')
    parser.add_argument('--channels', type=int, default=1, help='cifar channel is 3')
    return check_args(parser.parse_args())


# Checking arguments
def check_args(args):
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    # if args.dataset == 'cifar' or args.dataset == 'stl10':
    #     args.channels = 3
    # else:
    #     args.channels = 1
    args.cuda = True if args.cuda == 'True' else False
    return args
