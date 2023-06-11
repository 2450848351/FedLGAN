import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from torchvision import utils
import math


SAVE_PER_TIMES = 100

class LSTMGenerator(torch.nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.

    Args:
        in_dim: Input noise dimensionality  1*1
        out_dim: Output dimensionality      3*3
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """
    def __init__(self, in_dim, out_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Tanh())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)

        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs

class LSTMDiscriminator(torch.nn.Module):
    """An LSTM based discriminator. It expects a sequence as input and outputs a probability for each element.

    Args:
        in_dim: Input noise dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Inputs: sequence of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, 1)
    """
    def __init__(self, in_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())


    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, 1)
        return outputs

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=1, stride=2, padding=0))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        # print("生成器输入:",x.size())
        # for name, module in self.main_module.named_children():
        #     x = module(x)
        #     if isinstance(module, nn.ConvTranspose2d):
        #         print("11Upsample({}) : {}".format(name, x.shape))
        #     elif isinstance(module, nn.Conv2d):
        #         print("11Conv2d({}) : {}".format(name, x.shape))
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=2, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=2, stride=1, padding=0))


    def forward(self, x):
        # for name, module in self.main_module.named_children():
        #     x = module(x)
        #     if isinstance(module, nn.ConvTranspose2d):
        #         print("22Upsample({}) : {}".format(name, x.shape))
        #     elif isinstance(module, nn.Conv2d):
        #         print("22Conv2d({}) : {}".format(name, x.shape))
        x = self.main_module(x)
        return self.output(x)

        # x = self.main_module(x)
        # return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class WGAN_GP(object):
    def __init__(self, args):
        print("WGAN_GradientPenalty init model.")
        self.args = args
        self.G = LSTMGenerator(1,3)
        # self.G = Generator(args.channels)
        # self.D = Discriminator(args.channels)
        self.D = LSTMDiscriminator(3)
        self.C = args.channels

        # Check if cuda is available
        self.check_cuda(args.cuda)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 64

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.number_of_hydrology = 10

        self.generator_iters = args.generator_iters
        self.critic_iter = 5
        self.lambda_term = 10
        self.MSE = torch.nn.MSELoss()


    def mae_value(y_true, y_pred):
        n = len(y_true)
        mae = sum(np.abs(y_true - y_pred)) / n
        return mae

    def mape_value(y_true, y_pred):
        n = len(y_true)
        mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
        return mape

    def rmse_value(y_true, y_pred):
        n = len(y_true)
        mse = sum(np.square(y_true - y_pred)) / n
        rmse =math.sqrt(mse)
        return rmse

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False


    def train(self, train_loader,test_loader):
        self.t_begin = t.time()

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                hydrology = self.data.__next__()  #64*3*3
                hydrology = np.expand_dims(hydrology, axis=1)  #64*1*3*3
                hydrology = torch.from_numpy(hydrology)  # 这里的hydrology是DoubleTensor
                hydrology = hydrology.type(torch.FloatTensor)
                print('hydrology的size:',hydrology.size())

                if (hydrology.size()[0] != self.batch_size):# Check for batch to have full batch_size
                    continue

                z = torch.rand((self.batch_size, 100, 1, 1))

                hydrology, z = self.get_torch_variable(hydrology), self.get_torch_variable(z)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real hydrology
                d_loss_real = self.D(hydrology)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake hydrology
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))

                fake_hydrology = self.G(z)
                d_loss_fake = self.D(fake_hydrology)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty

                gradient_penalty = self.calculate_gradient_penalty(hydrology.data, fake_hydrology.data)
                # gradient_penalty.backward() 不支持双RNN反向传播，因为此时生成器和判别器都是lstm组成

                MSEloss = self.loss_func(fake_hydrology, hydrology)

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'Discriminator iteration: {d_iter}/{self.critic_iter}, MSEloss:{MSEloss}, d_loss: {d_loss}, Wasserstein_D: {Wasserstein_D}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake hydrology
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
            fake_hydrology = self.G(z)
            print('fake_hydrology:',fake_hydrology.size())
            g_loss = self.D(fake_hydrology)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
            # Saving model and sampling hydrology every 1000th generator iterations
            # if (g_iter) % SAVE_PER_TIMES == 0:
            #     self.save_model()
            #
            #     # Testing
            #     time = t.time() - self.t_begin
            #     print("Generator iter: {}".format(g_iter))
            #     print("Time {}".format(time))
        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # Save the trained parameters
        self.save_model()
        print("模型训练完毕,开始测试:")

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 hydrology saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')


    def calculate_gradient_penalty(self, real_hydrology, fake_hydrology):
        eta = torch.FloatTensor(self.batch_size,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_hydrology.size(1), real_hydrology.size(2))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta
        interpolated = eta * real_hydrology + ((1 - eta) * fake_hydrology)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_hydrology(self, hydrology, number_of_hydrology):
        if (self.C == 3):
            return self.to_np(hydrology.view(-1, self.C, 32, 32)[:self.number_of_hydrology])
        else:
            return self.to_np(hydrology.view(-1, 32, 32)[:self.number_of_hydrology])

    def generate_img(self, z, number_of_hydrology):
        samples = self.G(z).data.cpu().numpy()[:number_of_hydrology]
        generated_hydrology = []
        for sample in samples:
            if self.C == 3:
                generated_hydrology.append(sample.reshape(self.C, 32, 32))
            else:
                generated_hydrology.append(sample.reshape(32, 32))
        return generated_hydrology

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        modeldir = 'saved_model/' + self.args.model + '/' + self.args.dataset
        if not os.path.exists(modeldir + '/'):
            os.makedirs(modeldir + '/')
        torch.save(self.G.state_dict(), './'+modeldir+'/generator.pkl')
        torch.save(self.D.state_dict(), './'+modeldir+'/discriminator.pkl')
        print('Models save to ./'+modeldir+'/generator.pkl & ./'+modeldir+'/discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        modeldir = 'saved_model/' + self.args.model + '/' + self.args.dataset
        D_model_path = os.path.join(os.getcwd(), modeldir, D_model_filename)
        G_model_path = os.path.join(os.getcwd(), modeldir, G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, hydrology in enumerate(data_loader):
                yield hydrology

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_hydrology/'):
            os.makedirs('interpolated_hydrology/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        hydrology = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            hydrology.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(hydrology, nrow=number_int )
        utils.save_image(grid, 'interpolated_hydrology/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated hydrology.")
