import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from model import AutoEncoder
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../mnist/', type=str)
    parser.add_argument('--bs', default=256, type=int)  # batchSize
    parser.add_argument('--me', default=10, type=int)  # max epoch
    parser.add_argument('--lr', default=0.005, type=float) # learning rate
    parser.add_argument('--outDir', default='../experiments/', type=str)
    parser.add_argument('--nz', default=2, type=int)  # latent code dimension

    return parser.parse_args()

def np_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def test_dataset(traindata, trainset):
    print("training data size:", traindata.train_data.size()) # (60000, 28, 28)
    print("training labels size:", traindata.train_labels.size()) # (60000,)
    fig = plt.figure()
    plt.imshow(traindata.train_data[2].numpy(), cmap='gray')
    plt.title('%i' % traindata.train_labels[2])
    fig.savefig(os.path.join(opts.outDir, "test.png"))
    plt.close(fig)

    x, t = iter(trainset).next()
    print("trainset data size:", x.size()) # (batchSize, 1, 28, 28)
    print("trainset label size:", t.size()) # (batchSize,)


def plot_encdec(dec, ori, step, epoch):
    # visualize the reconstruction, comparing original vs reconstructed images
    if torch.cuda.is_available():
        dec, ori = dec.cpu(), ori.cpu()

    f = plt.figure()
    f, a = plt.subplots(2, 5, figsize=(5, 2))
    for i in range(5):
        # row with original images
        a[0][i].imshow(np.reshape(ori.data.numpy()[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(()); a[0][i].set_yticks(())

        # row with decoded images
        a[1][i].clear()
        a[1][i].imshow(np.reshape(dec.data.numpy()[i], (28, 28)), cmap='gray')
        a[1][i].set_xticks(()); a[1][i].set_yticks(())
    f.savefig(os.path.join(opts.outDir, "epoch{}_step{}.png".format(epoch,step)))
    plt.close(f)


def visualize(autoencoder, outDir, trainset, traindata):
    # get your latent space in 2D (should try t-SNE representation later on)
    autoencoder.eval()

    oriEx = np_var(traindata.train_data[:200]
                        .view(-1, 28*28).type(torch.FloatTensor)/255.)
    encEx, _ = autoencoder.forward(oriEx)

    if torch.cuda.is_available():
        encEx, oriEx = encEx.detach().cpu(), oriEx.detach().cpu()

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(encEx[:, 0], encEx[:, 1], c=traindata.train_labels[:200].numpy())
    plt.colorbar()
    plt.savefig(os.path.join(opts.outDir, "latentSpace.png"))
    plt.close(fig)


def train(autoencoder, outDir, trainset, traindata):

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=opts.lr)

    # original image to plot using plot_encdec
    oriEx = np_var(traindata.train_data[:5]
                        .view(-1, 28*28).type(torch.FloatTensor)/255.)

    for epoch in range(opts.me):
        for step, (x, y) in enumerate(trainset):
            autoencoder.train()
            x = np_var(x.view(-1, 28*28))   # batch x, shape (batch, 28*28)
            y = np_var(y)                   # batch label

            z, rec = autoencoder.forward(x)

            loss = autoencoder.lossfunc(rec, x)  # mean square error

            optimizer.zero_grad()   # clear gradients for this training step
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

                # plotting encoded/decoded image
                _, decEx = autoencoder.forward(oriEx)
                plot_encdec(dec=decEx, ori=oriEx, step=step, epoch=epoch)



if __name__ == '__main__':

    opts = get_args()

    if not os.path.exists(opts.outDir):
        os.makedirs(opts.outDir)

    # mnist digits dataset
    transforms_ = torchvision.transforms.ToTensor()
    traindata = torchvision.datasets.MNIST(root=opts.root, train=True,
                 transform=transforms_, download=True)
    trainset = DataLoader(dataset=traindata, batch_size=opts.bs, shuffle=False)

    # getting the structure of your autoencoder
    autoencoder = AutoEncoder(opts.nz)

    # useful when you have a GPU (link pytorch to CUDA)
    if torch.cuda.is_available():
       autoencoder.cuda()

    print("test the data")
    test_dataset(traindata=traindata, trainset=trainset)

    print("begin training ...")
    train(autoencoder=autoencoder, outDir=opts.outDir, trainset=trainset,
          traindata=traindata)

    print("visualize latent space representation")
    visualize(autoencoder=autoencoder, outDir=opts.outDir, trainset=trainset,
              traindata=traindata)
