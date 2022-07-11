#  Experiences on VAE
#  Robert Martí robert.marti@ugd.edu
#  Based on https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71

"""
The following is an import of PyTorch libraries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import os


"""
Determine if any GPUs are available
"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=32*120*120, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, 32*120*120)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 120, 120)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar



def main():

    """
    Initialize Hyperparameters
    """
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 10 # was 
    
    dataset = "mammo" # you can choose mammo o mnist


    """
    Create dataloaders to feed data into the neural network
    Default MNIST dataset is used and standard train/test split is performed
    """
    if (dataset == "mnist"):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                        transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
            batch_size=1)
    if (dataset == "mammo"):
        batch_size = 128
        num_epochs = 50 # was 

        path = "/mnt/mia_images/breast/iceberg_selection2/HOLOGIC/roi"
        print ("Mammo - Batch Size ", batch_size)
        # transform=transforms.ToTensor()
        transform=transforms.Compose([transforms.Resize(size=(128, 128)),
            transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
        dataset = ImageFolder(root=path, transform=transform)   
        n = len(dataset)  # total number of examples
        n_test = int(0.1 * n)  # take ~10% for test
        test_set = torch.utils.data.Subset(dataset, range(n_test))  # take first 10%
        train_set = torch.utils.data.Subset(dataset, range(n_test, n))  # take the rest 

        # transform = T.Compose([T.Resize((150, 200)),T.ToTensor()])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        
        # path = "/mnt/mia_images/breast/iceberg_selection2/HOLOGIC/roi"
        # test_loader = DataLoader(dataset, train = False, batch_size=batch_size, shuffle=True)
        

# for batch_number, (images, labels) in enumerate(dataloader):
# print(batch_number, labels)
# print(images)
# break


    """
    Initialize the network and the Adam optimizer
    """
    net = VAE().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


    """
    Training the network for a given number of epochs
    The loss after every epoch is printed
    """
    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader, 0):
            imgs, _ = data
            #print(imgs.size())
            imgs = imgs.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = net(imgs)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {}: Loss {}'.format(epoch, loss))

    # Save network
    torch.save(net, "testVAE_mammo.pth")

if __name__ == "__main__":
    main()