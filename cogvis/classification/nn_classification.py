import copy
import functools
import multiprocessing
import time
import warnings

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


class HDF5Dataset(utils.data.Dataset):

    def __init__(self, filename, transform=None):
        super(HDF5Dataset, self).__init__()
        self.data = h5py.File(filename, 'r')
        self.classes, self.class_to_idx = self._find_classes(self.data)
        self.idx_to_sample_idx = self._idx_mapping(self.data)
        self.transform = transform

    def _find_classes(self, data):
        classes = [cls_name for cls_name in data]
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _idx_mapping(self, data):
        sample_idxs = []
        for class_name in data:
            for sample_idx in range(len(data[class_name])):
                sample_idxs.append((class_name, sample_idx))
        sample_idxs.sort(key=lambda x: x[1]) # ensures locality of reference
        idx_to_sample_idx = {idx: s for idx, s in enumerate(sample_idxs)}
        return idx_to_sample_idx

    def __len__(self):
        return len(self.idx_to_sample_idx)

    def __getitem__(self, idx):
        sample_idx = self.idx_to_sample_idx[idx]
        class_name, instance_idx = sample_idx 
        instance = self.data[class_name][instance_idx]
        target = self.class_to_idx[class_name]
        if self.transform:
            instance = self.transform(instance)
        return instance, target 
        

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


def existing(model_name, num_classes, pretrained=False, feature_extract=False):
    model =  eval(f'models.{model_name}(pretrained={pretrained})')
    _requires_grad(model, feature_extract)
    if 'resnet' in model_name:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif 'alexnet' in model_name or 'vgg' in model_name:
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif 'squeezenet' in model_name:
        in_channels = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(in_channels, num_classes, 
                kernel_size=(1,1), stride=(1,1))
    elif 'densenet' in model_name:
        in_features = model.classifier[1].in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise NotImplementedError(f'{model_name} not supported')
    params = [p for p in model.parameters() if p.requires_grad==True]
    return model, params


def _requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def data_loader(path, batch_size, chunking=False, shuffle=True, num_workers=0, 
        transform=None): 
    # NOTE: see PyTorch's documentation:
    # some transforms work for only PIL images and some only numpy.ndarray
    if path.endswith('.hdf5'):
        if num_workers > 0:
            msg = 'HDF5 dataset implementation does not support'\
                    'multi-threaded data access. Try num_workers=0'
            raise ValueError(msg)
        if chunking and shuffle:
            msg = 'Shuffle negates the effect of chunking. Try shuffle=False'
            warnings.warn(msg, RuntimeWarning)
        dataset = HDF5Dataset(path, transform=transform)
    else:
        if chunking:
            msg = 'ImageFolder does not support chunking. Try chunking=False'
            warnings.warn(msg, RuntimeWarning)
        dataset = datasets.ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)
    return len(dataset), loader


def data_transform(**kwargs):
    data_transforms = []
    for transform_name, args in kwargs.items():
        expr = f'transforms.{transform_name}{args}'
        data_transforms.append(eval(expr))
    return transforms.Compose(data_transforms)


def train(model, train_loader, val_loader, train_size, val_size, criterion, 
        optimizer, scheduler, num_epochs, device='cpu'):
    device = torch.device('cuda:0' if device == 'gpu' else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}/{num_epochs-1}')
        print('-' * 10)
        epoch_loss, epoch_acc = epoch_train(model, train_loader, train_size, 
                criterion, optimizer, scheduler, device)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        epoch_loss, epoch_acc = epoch_eval(model, val_loader, val_size, 
                criterion, device)
        print(f'Val. Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print()
    print(f'Best Val. Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    if device == 'gpu':
        model = model.to(torch.device('cpu'))
    return model


def epoch_train(model, loader, size, criterion, optimizer, scheduler, device):
    model.train()  
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in loader: 
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    scheduler.step()
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.double() / size 
    return epoch_loss, epoch_acc


def epoch_eval(model, loader, size, criterion, device):
    model.eval()  
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in loader: 
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.double() / size 
    return epoch_loss, epoch_acc


#if __name__ == '__main__':
#    dataset = HDF5Dataset('train.hdf5')
#    print(dataset.__getitem__(7500))


if __name__ == '__main__':
    train_dir = 'orchard_data/train/'
    val_dir = 'orchard_data/val/'
    test_dir = 'orchard_data/test/'

    train_dir = 'train.hdf5'
    val_dir = 'val.hdf5'
    # ----- 1 ----- #
    transform = data_transform(
            ToTensor=(),
            )
    #transform = data_transform(
    #        Resize=(70,), 
    #        CenterCrop=(64,),
    #        ToTensor=(),
    #        )
    size, loader = data_loader(train_dir, 1000, transform=transform)
    data, _ = next(iter(loader))
    means = [data[:,c].mean().tolist() for c in range(len(data[0]))] 
    stds = [data[:,c].std().tolist() for c in range(len(data[0]))]

    # ----- 2 ----- #
    batch_size = 32
    transform = data_transform(
        ToTensor=(),
        Normalize=(means, stds),
        )
    #transform = data_transform(
    #    Resize=(70,),
    #    CenterCrop=(64,),
    #    ToTensor=(),
    #    Normalize=(means, stds),
    #    )
    train_size, train_loader = data_loader(train_dir, batch_size, 
            transform=transform)
    val_size, val_loader = data_loader(val_dir, batch_size, transform=transform)

    # ----- 3 ----- #   
    #nn_model = nn_classifier.MLP([150528, 250, 2])
    #nn_model = nn_classifier.CNN([3, 6, 16], [16 * 53 * 53, 120, 84, 2], conv_ks=5, pool_ks=2)
    #nn_model = nn_classifier.existing_model('resnet18', 2)
    nn_model, params_to_update = existing('squeezenet1_1', 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    start = time.perf_counter()
    nn_model = train(
            nn_model, 
            train_loader, val_loader, 
            train_size, val_size, 
            criterion, 
            optimizer,
            exp_lr_scheduler, 
            num_epochs=2,
            )
    end = time.perf_counter()
    print(f'Time: {end - start} s')





    
