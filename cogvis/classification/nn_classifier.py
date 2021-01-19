import multiprocessing

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class NeuralNetClassifier(nn.Module):

    def __init__(self):
        pass

    def get_data_loader(self, data_dir, batch_size, **transform_params):
        data_transform = self._transform(transform_params)
        dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
        num_procs = multiprocessing.cpu_count() - 1
        return DataLoader(dataset, batch_size, shuffle=True, 
                num_workers=num_procs)

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
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        assert hidden_sizes, "hidden_sizes is empty" 
        self.input_fc = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_fc = nn.ModuleList()
        for k in range(len(hidden_sizes)-1):
            self.hidden_fc.append(nn.Linear(hidden_sizes[k], hidden_sizes[k+1]))
        self.output_fc = nn.Linear(input_size, hidden_sizes[-1])

    def forward(self):
        x = F.relu(self.input_fc)
        for l in hidden_fc:
            x = F.relu(l)
        x = F.relu(self.output_fc)
        return x




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


