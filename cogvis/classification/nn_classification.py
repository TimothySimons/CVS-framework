"""
Module for loading, transforming and classifying images using neural networks.
"""

import copy
import functools
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
    """Provides access to images stored in an HDF5 file.

    .. note::
        This class shouldn't be directly instantiated by the user but rather 
        indirectly through the :meth:`data_loader` method.
    """

    def __init__(self, filename, transform=None):
        """Initialize self.

        :type filename: str
        :param filename: 
            Specifies the path to the HDF5 file containing the dataset.
        :type transform: Transform object, optional
        :param transform: 
            The composed or single transform to be applied to the loaded data.
        """
        super(HDF5Dataset, self).__init__()
        self.data = h5py.File(filename, 'r')
        self.classes, self.class_to_idx = self._find_classes(self.data)
        self.idx_to_sample_idx = self._idx_mapping(self.data)
        self.transform = transform

    def _find_classes(self, data):
        """Returns class names and class index mapping of the loaded data."""
        classes = [cls_name for cls_name in data]
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _idx_mapping(self, data):
        """Returns an index mapping to all images in the HDF5 database."""
        sample_idxs = []
        for class_name in data:
            for sample_idx in range(len(data[class_name])):
                sample_idxs.append((class_name, sample_idx))
        sample_idxs.sort(key=lambda x: x[1]) # ensures locality of reference
        idx_to_sample_idx = {idx: s for idx, s in enumerate(sample_idxs)}
        return idx_to_sample_idx

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.idx_to_sample_idx)

    def __getitem__(self, idx):
        """Returns a sample of the dataset."""
        sample_idx = self.idx_to_sample_idx[idx]
        class_name, instance_idx = sample_idx 
        instance = self.data[class_name][instance_idx]
        target = self.class_to_idx[class_name]
        if self.transform:
            instance = self.transform(instance)
        return instance, target 
        

class MLP(nn.Module):
    """A multi-layer perceptron (feed-forward neural network) model."""
    def __init__(self, dims):
        """Initializes self.

        :type dims: list object
        :param dims:
            Specifies the number of perceptrons in each layer of the MLP.
        """
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(dims)-1):
            self.linears.append(nn.Linear(dims[i], dims[i+1]))

    def forward(self, x):
        """Computes the forward pass of the neural network."""
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        for i in range(len(self.linears)-1):
            x = self.linears[i](x)
            x = F.relu(x)
        return F.softmax(self.linears[-1](x), dim=1)


class CNN(nn.Module):
    """A convolutional neural network model."""
    def __init__(self, conv_dims, linear_dims, conv_ks, pool_ks):
        """Initializes self.

        :type conv_dims: list 
        :param conv_dims:
            The first item is the number of channels of the image. Subsequent
            items specify the number of filters for each subsequent convolution.
        :type linear_dims: list 
        :param linear_dims:
            Specifies the number of perceptrons in the feedforward part of the 
            CNN.
        :type conv_ks: int 
        :param conv_ks:
            The convolution kernel size.
        :type pool_ks: int
        :param pool_ks:
            The pooling kernel size.
        """
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
        """Computes the forward pass of the neural network."""
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
    """
    Returns a pre-built model modified to produce an output of size 
    *num_classes*.

    :type model_name: str 
    :param model_name: 
        The name of the existing model. Currently, the supported models are
        all versions of *resnet*, *alexnet*, *squeezenet* and *densenet*. See 
        the `PyTorch documentation 
        <https://pytorch.org/vision/0.8/models.html>`_ for version details.
    :type num_classes: int
    :param num_classes: 
        The number of classes the model must predict. The architecture of 
        the chosen model is modified to produce these class predictions.
    :type pretrained: bool, default False
    :param pretrained:
        If True, returns an existing model that has been trained on 
        ImageNet.
    :type feature_extract: bool, default False
    :param feature_extract:
        If True, returns a model in which the pre-trained model parameters
        are frozen and are thus no longer updated during training. The 
        existing model thereby acts as a feature extractor whose output are 
        the inputs to the modified/added layers of the returned model.

    :raises NotImplementedError:
        Raised when the specified PyTorch model is not supported by cogvis.

    :returns:
        A neural network model
    """
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
    """Freezes model parameters."""
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def data_loader(path, batch_size, chunking=False, shuffle=True, num_workers=0, 
        transform=None): 
    """Returns the size of the dataset and a DataLoader object.

    :type path: str
    :param path:
        The path to the dataset. The path either points to a directory 
        containing class folders and corresponding images (jpg, png, etc.)
        or to an HDF5 file containing the relevant classes and image data.
    :type batch_size: int
    :param batch_size:
        The number of items to be retrieved from the data loader at a time.
    :type chunking: bool, default False
    :param chunking:
        If True, data loader assumes the dataset is a chunked HDF5 dataset. 
        This will ensure locality of reference is maintained when retrieving 
        data from the dataset.
    :type shuffle: bool, default True
    :param shuffle: 
        If True, the data will be retrieved randomly from the dataset.
    :type num_workers: int, optional
    :param num_workers:
        Specifies the number of CPU workers to load images. This only 
        applies when using an ImageFolder dataset.

    :raises ValueError: 
        Raised when using an HDF5 dataset and `num_workers` > 0. The HDF5 
        dataset does not support multi-threaded data access.

    :returns:
        DataLoader for reading images from memory.
    """
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
    """Chains together the chosen transforms.

    .. note::
        Transforms to improve the robustness of the model, such as 
        ``RandomPerspective`` or ``RandomRotation``, should only be applied to 
        the training data. Other transforms, such as ``Resize`` or 
        ``CenterCrop`` can be applied to train, validation and test datasets.

    .. warning::
        Some transforms work for only ``PIL`` images and others only 
        ``numpy.ndarray``. Be aware of the type of dataset you are using when 
        specifying transforms. See the `PyTorch documentation 
        <https://pytorch.org/vision/0.8/transforms.html>`_ for more details.

    :returns: 
        A composition of transforms.

    **Example:**

    .. code-block:: python

        transform = nn_classifier.data_transform(
        Resize = (256,),
        CenterCrop= (224,),
        ToTensor = (),
        Normalize = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    """
    data_transforms = []
    for transform_name, args in kwargs.items():
        expr = f'transforms.{transform_name}{args}'
        data_transforms.append(eval(expr))
    return transforms.Compose(data_transforms)


def train(model, train_loader, val_loader, train_size, val_size, criterion, 
        optimizer, scheduler, num_epochs, device='cpu'):
    """
    Trains and validates the model on the train and validation datasets using 
    the chosen criterion, optimizer and schedular.

    :type model: torch.nn.Module object
    :param model: The model to be trained.
    :type train_loader: torch.util.data.DataLoader object
    :param train_loader: A data loader to read in batches of training input images.
    :type val_loader: torch.util.data.DataLoader object
    :param val_loader: 
        A data loader to read in batches of validation input images.
    :type train_size: int
    :param train_size:
        The size of the training dataset.
    :type val_size: int
    :param val_size:
        The size of the validation dataset.
    :type criterion: torch.nn.Loss object
    :param criterion: 
        A criterion measures the loss between the target and the output of the 
        model.
    :type optimizer: torch.optim.Optimizer object
    :param optimizer:
        An optimizer holds the current state of the model updates the model 
        parameters based on the computed gradients.
    :type schedular: torch.optim.LRSchedular object
    :param schedular:
        The learning rate schedular adjusts the learning rate based on the 
        number of epochs.
    :type num_epochs: int
    :param num_epochs: 
        The number of epochs to train the model for.
    :type device: str, default 'cpu'
    :param device:
        The device ('cpu' or 'gpu') on which train the model.

    :returns:
        The trained model.
    """
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
    """Trains the model for a single epoch.

    .. seealso:: 
        .. py:method:: train 
            :noindex:

    :returns:
        The loss and accuracy of the model after epoch.
    """
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
    """
    Evaluates the model on a dataset for a given epoch.

    .. seealso:: 
        .. py:method:: evaluate
            :noindex:

    :returns:
        The loss and accuracy of the model. 
    """
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


def evaluate(model, loader, size, criterion, device='cpu'):
    """
    Calculates loss and accuracy metrics of the model in predicting classes of a 
    dataset retrived from the *loader*.
    
    :type model: torch.nn.Module object
    :param model: 
        The model to be evaluated.
    :type train_loader: torch.util.data.DataLoader object
    :param train_loader: 
        A data loader to read in batches of input images.
    :type size: int
    :param size:
        The size of the dataset.
    :type criterion: torch.nn.Loss object
    :param criterion: 
        A criterion measures the loss between the target and the output of the 
        model.
    :type device: str, default 'cpu'
    :param device:
        The device ('cpu' or 'gpu') on which evaluate the model.

    :returns: 
        A tuple containing loss and accuracy metrics.
    """
    device = torch.device('cuda:0' if device == 'gpu' else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    loss, acc = epoch_eval(model, loader, size, criterion, device)
    print(f'Loss: {loss:.4f} Acc: {acc:.4f}')

