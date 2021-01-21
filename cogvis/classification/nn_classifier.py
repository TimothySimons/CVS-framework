import copy
import functools
import multiprocessing
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


class MLP(nn.Module):

    def __init__(self, dims):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(dims)-1):
            self.linears.append(nn.Linear(dims[i], dims[i+1]))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        for i in range(len(self.linears)-1):
            x = self.linears[i](x)
            x = F.relu(x)
        return F.softmax(self.linears[-1](x), dim=1)


class CNN(nn.Module):

    def __init__(self, conv_dims, linear_dims, kernel_size):
        # TODO: need to come up with a smarter way of passing func args
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList()
        for i in range(len(conv_dims)-1):
            conv = nn.Conv2d(conv_dims[i], conv_dims[i+1], kernel_size)
            self.convs.append(conv)
        self.linears = nn.ModuleList()
        for i in range(len(linear_dims)-1):
            linear = nn.Linear(linear_dims[i], linear_dims[i+1])
            self.linears.append(linear)

    def forward(self, x):
        batch_size = x.shape[0]
        for i in range(len(self.convs)):
            x = self.convs[i](x) 
            x = F.max_pool2d(x, kernel_size=2) #TODO: need to pass in ks
        x = x.view(batch_size, -1)
        for i in range(len(self.linears)-1):
            x = self.linears[i](x)
            x = F.relu(x)
        return F.softmax(self.linears[-1](x), dim=1)


def existing_model(model_name, out_size, **kwargs):
    kwargs_str = ''.join(f'{k}={v},' for k, v in kwargs.items())
    model =  eval(f'models.{model_name}({kwargs_str})')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, out_size)
    return model


def data_loader(data_dir, batch_size, data_transform): 
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    num_procs = multiprocessing.cpu_count() - 1
    return DataLoader(dataset, batch_size, shuffle=True, 
            num_workers=num_procs)


def data_transform(**transforms_args):
    data_transforms = []
    for transform_name, kwargs in transform_args.items():
        kwargs_str = ''.join(f'{k}={v},' for k, v in kwargs.items())
        expr = f"transforms.{transform_name}({kwargs_str})"
        data_transforms.append(eval(expr))
    return transforms.Compose(data_transforms)


def train_classifier(model, data_loaders, dataset_sizes, criterion, 
        optimizer, scheduler, num_epochs=2): # TODO: need to change how we do params
    device = 'cpu'
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__== '__main__':
    # Just ignore this - random testing

    #nn_model = MLP([784, 250, 100, 10])

    #data_dir = 'hymenoptera_data/train'
    #train_data_loader = nn_model.get_data_loader(data_dir,
    #        batch_size=4,
    #        RandomResizedCrop=(224,),
    #        RandomHorizontalFlip=(),
    #        ToTensor=(),
    #        Normalize=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    #data_dir = 'hymenoptera_data/val'
    #val_data_loader = nn_model.get_data_loader(data_dir,
    #        batch_size=4,
    #        Resize=(256,),
    #        CenterCrop=(224,),
    #        ToTensor=(),
    #        Normalize=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    #data_loaders = {'train': train_data_loader, 'val': val_data_loader}
    #optimizer = optim.Adam(nn_model.parameters())
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #criterion = nn.CrossEntropyLoss()
    # nn_model.train_classifier(data_loaders, criterion, optimizer, exp_lr_scheduler)
    model_ft = existing_model('resnet18', 2, pretrained=True)


    nn_model = CNN([1, 6, 16], [256, 120, 84, 10], kernel_size=5)
    ROOT = '.data'
    train_data = datasets.MNIST(root = ROOT, train = True, download = True)
    mean = train_data.data.float().mean() / 255
    std = train_data.data.float().std() / 255 

    train_transforms = transforms.Compose([
        transforms.RandomRotation(5, fill=(0,)),
        transforms.RandomCrop(28, padding = 2),
        transforms.ToTensor(),
        transforms.Normalize(mean = [mean], std = [std])
        ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [mean], std = [std])
        ])

    train_data = datasets.MNIST(root = ROOT, 
            train = True, 
            download = True, 
            transform = train_transforms)

    test_data = datasets.MNIST(root = ROOT, 
            train = False, 
            download = True, 
            transform = test_transforms)

    VALID_RATIO = 0.9

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms


    BATCH_SIZE = 5 

    train_iterator = data.DataLoader(train_data, 
            shuffle = True, 
            batch_size = BATCH_SIZE)

    valid_iterator = data.DataLoader(valid_data, 
            batch_size = BATCH_SIZE)

    test_iterator = data.DataLoader(test_data, 
            batch_size = BATCH_SIZE)

    data_loaders = {'train': train_iterator, 'val': valid_iterator}
    dataset_sizes = {'train': n_train_examples, 'val': n_valid_examples}

    optimizer = optim.Adam(nn_model.parameters())
    criterion = nn.CrossEntropyLoss()
    #nn_model.train_mnist(train_iterator, optimizer, criterion)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #train_classifier(nn_model, data_loaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler)



