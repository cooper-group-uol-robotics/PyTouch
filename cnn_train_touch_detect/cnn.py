import os

import torch
import torch.nn as nn
import numpy as np
import random
import argparse

from PIL import Image
from matplotlib import pyplot as plt
from torch import optim
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models

from data_process import load_data
from models import initialise_vision_model
from pytorchtools import EarlyStopping

from train.models.touch_detect import TouchDetectionModel

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([('polynomial_features', polynomial_features), ("linear_regression", linear_regression)])
    return pipeline


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = cnn().to(device)

SUPPORTED_MODELS = ["resnet18", "alexnet", "vgg", "squeezenet", "densenet", "inception"]
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=SUPPORTED_MODELS, help='The name of the supported classifier model to use.')
parser.add_argument('--mode', choices=['finetune', 'featureextract'],
                    help='The modality of the trained model. Choose between finetune and featureextract')
args = parser.parse_args()

# MODEL = args.model
MODEL = 'resnet18'
# MODE = args.mode
MODE = 'finetune'


def load_cnn_model(MODEL, mode):
    if mode == 'featureextract':
        FEAT_EXT = True
    else:
        FEAT_EXT = False
    model = initialise_vision_model(model_name=MODEL, num_classes=2, feature_extract=FEAT_EXT)

    return model


model = load_cnn_model(MODEL, MODE)
model = model.to(device=device)

optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss().to(device)


def train(patience, n_epochs, train_loader, valid_loader):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print('train...')

    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if batch_idx % 10 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx / len(train_loader),
                                                                         loss.item()))
        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            # if epoch == 2 or epoch == 60:
            #     X = data
            #     y = target
            #     cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            #     titles = ['Learning Curves (Under Fitting)', 'Learning Curves', 'Learning Curves (Over Fitting)']
            #     degrees = [1, 3, 10]
            #
            #     plt.figure(figsize=(18, 4), dpi=200)
            #     for i in range(len(degrees)):
            #         plt.subplot(1, 3, i + 1)
            #         plot_learning_curve(polynomial_model(degrees[i]), titles[i], X, y, ylim=(0.75, 1.01), cv=cv)
            #
            #     plt.show()

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        rain_losses = []
        alid_losses = []

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    torch.save(model.state_dict(), "model/cnn_resnet18.pkl")

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('model/cnn_resnet18.pkl'))
    model.eval()

    return model, avg_train_losses, avg_valid_losses


to_pil_image = transforms.ToPILImage()

if __name__ == '__main__':
    batch_size = 4
    n_epochs = 100
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 2
    train_loader, valid_loader = load_data(batch_size)
    model, train_loss, valid_loss = train(patience, n_epochs, train_loader, valid_loader)

    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of the lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')
