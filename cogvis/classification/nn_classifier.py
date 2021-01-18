import multiprocessing

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class NeuralNetClassifier:

    def __init__(self):
        pass

    def load(self, data_dir, batch_size, **transform_params):
        data_transform = self._transform(transform_params)
        dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
        num_procs = multiprocessing.cpu_count() - 1
        #inputs, classes = next(iter(data_loader))
        #print(inputs, classes)
        return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_procs)

    def _transform(self, transform_params):
        data_transforms = []
        for k, v in transform_params.items():
            expr = f"transforms.{k}{v}"
            data_transforms.append(eval(expr))
        return transforms.Compose(data_transforms)


    def train(self, X, y):
        pass

    def predict(self, X):
        pass


class MLP(NeuralNetClassifier):
    def __init__(self):
        super().__init__()


if __name__== '__main__':
    # Just ignore this - random testing
    from os import listdir
    from os.path import isfile, join
    from PIL import Image
    class_dir = 'hymenoptera_data/train'
    #image_data = []
    #for filename in listdir(data_dir):
    #    filename = join(data_dir, filename)
    #    image = Image.open(filename)
    #    image_data.append(np.asarray(image))
    #print(image_data[0].shape)
    #print(image_data[1].shape)
    model = MLP()
    model.load(class_dir, 4, RandomResizedCrop=(224,), ToTensor=(), Normalize = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))


