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
    # TODO: allow them to choose more parameters, like stride etc.

    def __init__(self, conv_dims, linear_dims, conv_ks, pool_ks):
        super(CNN, self).__init__()
        self.pool_ks = pool_ks
        self.convs = nn.ModuleList()
        for i in range(len(conv_dims)-1):
            conv = nn.Conv2d(conv_dims[i], conv_dims[i+1], conv_ks) 
            self.convs.append(conv)
        self.linears = nn.ModuleList()
        for i in range(len(linear_dims)-1):
            linear = nn.Linear(linear_dims[i], linear_dims[i+1])
            self.linears.append(linear)

    def forward(self, x):
        batch_size = x.shape[0]
        for i in range(len(self.convs)):
            x = self.convs[i](x) 
            x = F.max_pool2d(x, kernel_size=self.pool_ks)
        x = x.view(batch_size, -1)
        for i in range(len(self.linears)-1):
            x = self.linears[i](x)
            x = F.relu(x)
        return F.softmax(self.linears[-1](x), dim=1)


def existing_model(model_name, out_size, *args, **kwargs):
    args_str = ''.join(f'{a},' for a in args)
    kwargs_str = args_str.join(f'{k}={v},' for k, v in kwargs.items())
    model =  eval(f'models.{model_name}({kwargs_str})')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, out_size)
    return model


def data_loader(data_dir, batch_size, data_transform): 
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    num_procs = multiprocessing.cpu_count() - 1
    loader = DataLoader(dataset, batch_size, shuffle=True, 
            num_workers=num_procs)
    return len(dataset), loader


def data_transform(**kwargs):
    data_transforms = []
    for transform_name, args in kwargs.items():
        expr = f"transforms.{transform_name}{args}"
        data_transforms.append(eval(expr))
    return transforms.Compose(data_transforms)


def train(model, train_loader, val_loader, train_size, val_size, criterion, 
        optimizer, scheduler, num_epochs, device='cpu'):

    data_loaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': train_size, 'val': val_size}
    device = torch.device('cuda:0' if device == 'gpu' else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    if device == 'gpu':
        model = model.to(torch.device('cpu'))
    return model


if __name__== '__main__':

    # Just ignore this - random testing

    train_transform = data_transform(
        Resize = (256,),
        CenterCrop= (224,),
        ToTensor = (),
        Normalize = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

    val_transform = data_transform(
        Resize = (256,),
        CenterCrop = (224,),
        ToTensor = (), 
        Normalize = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )

    train_dir = 'hymenoptera_data/train/'
    val_dir = 'hymenoptera_data/val/'
    train_size, train_loader = data_loader(train_dir, 4, train_transform)
    val_size, val_loader = data_loader(val_dir, 4, val_transform)

    print(train_size, val_size)

    #nn_model = MLP([150528, 250, 2])
    #nn_model = CNN([3, 6, 16], [16 * 53 * 53, 120, 84, 2], conv_ks=5, pool_ks=2)
    nn_model = existing_model('resnet18', 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(nn_model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    nn_model = train(
            nn_model, 
            train_loader, val_loader, 
            train_size, val_size, 
            criterion, 
            optimizer, 
            exp_lr_scheduler, 
            num_epochs=5
            )







