import torch
from torch.autograd import Variable
import torch.nn.functional as F
from settings import *
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
SCALE = 255.0
IMAGE_SIZE = 28 * 28
K = 1
EPOCH = 100
FAKE_IMG_SIZE = 256

def get_data_mnist_loader(train=True):
    return torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_DATA,
                       train=train,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),

        batch_size=BATCH_SIZE,
        shuffle=True)


class GeneratorNet(nn.Module):
    def __init__(self, input_size=IMAGE_SIZE, output_size=IMAGE_SIZE, layers=(50, 50), scale=SCALE):
        super(GeneratorNet, self).__init__()

        # self.layer_dims = np.concatenate(([input_size], list(layers), [output_size]))
        # self.fcs = [nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]) for i in range(len(self.layer_dims) - 1)]
        # self.scale = SCALE
        self.fc1 = nn.Linear(FAKE_IMG_SIZE, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, IMAGE_SIZE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # for i in range(len(self.fcs) - 1):
        x = Variable(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5)
        x = self.fc4(x)

        x = self.sigmoid(x)

        # x = torch.mul(x, SCALE)

        # x = x.view(-1, 1, 28, 28)
        return x


class DiscriminatorNet(nn.Module):
    def __init__(self, input_size=IMAGE_SIZE, output_size=1, layers_cnn=(32, 20), kernel=3, layers_dnn=(320, 50)):
        super(DiscriminatorNet, self).__init__()

        # self.conv_dims = np.concatenate(([input_size], list(layers_cnn)))
        # self.fc_dims = np.concatenate(([list(layers_dnn), [output_size]]))
        #
        # self.convs = [nn.Conv2d(i, i + 1, kernel_size=kernel) for i in range(len(self.fc_dims) - 1)]
        # self.fcs = [nn.Linear(self.fc_dims[i], self.fc_dims[i + 1]) for i in range(len(self.fc_dims) - 1)]
        # self.conv1 = nn.Conv2d(1, 5, kernel_size=kernel, padding=1)
        # self.conv2 = nn.Conv2d(5, 5, kernel_size=kernel, padding=1)
        #
        # self.fc1 = nn.Linear(5*IMAGE_SIZE, 1)
        # self.fc2 = nn.Linear(layers_dnn[1], output_size)

        self.fc1 = nn.Linear(IMAGE_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # for conv in self.convs:
        #     x = F.relu(conv(x))
        #
        # x = x.view(-1, self.fc_dims[0])
        #
        # for i in range(len(self.fcs) - 1):
        #     x = F.relu(self.fcs[i](x))
        #
        # x = self.fcs[-1](x)
        # print(x.shape)
        # x = F.relu(self.conv1(x))
        #
        # x = F.relu(self.conv2(x))
        #
        # x = x.view(-1, 5*IMAGE_SIZE) # hack
        #
        # x = self.fc1(x)
        x = x.view(-1, IMAGE_SIZE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        x = self.sigmoid(x)

        return x





PRINT = False

def get_next_batch(data_loader, epoch=EPOCH):
    for ep in range(epoch):
        for batch_idx, (data, _) in enumerate(data_loader):
            yield ep, data.cuda()
        global PRINT

        PRINT = True
    yield None


def train(train_loader):
    gen_model = GeneratorNet()
    gen_model.cuda()
    gen_optimizer = optim.Adam(gen_model.parameters(), lr=0.000002)

    disc_model = DiscriminatorNet()
    disc_model.cuda()
    disc_optimizer = optim.Adam(disc_model.parameters(), lr=0.0002)

    batch_data = get_next_batch(train_loader, EPOCH)

    train_loop = True
    while(train_loop):
        gen_model.eval()
        disc_model.train()
        for i in range(K):
            try:
                _, data = next(batch_data)
            except StopIteration:
                train_loop = False
                break
            # for batch_idx, (data, _) in enumerate(train_loader):
            data = Variable(data)
            disc_optimizer.zero_grad()
            noise_data = torch.rand(data.size(0), FAKE_IMG_SIZE).cuda()

            y_noise_data = Variable(torch.zeros(data.size(0)).cuda())
            y_data = Variable(torch.ones(data.size(0)).cuda())

            loss_d_data = torch.nn.BCELoss()
            loss_d_noise_data = torch.nn.BCELoss()

            g_d = gen_model(noise_data)
            d_nd = disc_model(g_d)
            d_d = disc_model(data)

            loss_d = loss_d_data(d_d, y_data) + loss_d_noise_data(d_nd, y_noise_data)

            loss_d.backward()
            disc_optimizer.step()
            # print('Discriminator loss: %f' % loss_d)
        try:
            ep, data = next(batch_data)
        except StopIteration:
            train_loop = False
            break

        disc_model.eval()
        gen_model.train()
        gen_optimizer.zero_grad()

        y_data = Variable(torch.ones(data.size(0)).cuda())

        noise_data = torch.rand(data.size(0), FAKE_IMG_SIZE).cuda()  # hacky

        g_d = gen_model(noise_data)
        d_nd = disc_model(g_d)

        loss_g = torch.nn.BCELoss()
        loss_g = loss_g(d_nd, y_data)
        loss_g.backward()
        gen_optimizer.step()
        # print()
        # print('Generator loss: %f' % loss_g)
        global PRINT
        if PRINT:
            PRINT = False
            print('Discriminator loss: %f' % loss_d)
            print('Generator loss: %f' % loss_g)
            plt.imshow(g_d.cpu().data.numpy()[-1].reshape((28,28)), cmap='gray')
            plt.savefig(os.path.join(PLOT_DIR, 'plot_' + str(ep)))


def main():
    train_loader = get_data_mnist_loader(train=True)
    train(train_loader)


if __name__ == '__main__':
    main()
