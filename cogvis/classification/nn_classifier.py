import torch
from torchvision import transforms

class NeuralNetClassifier:
    def __init__(self):
        pass

    def transform(self, mean, std, **transform_params):
        # TODO: doc - must pass in the class of transform as the key and the 
        # params for that class as the value.
        data_transforms = []
        for k, v in transform_params.items():
            expr = f"transforms.{k}({v if v is not None else ''})"
            data_transforms.append(eval(expr))
        data_transforms.append(transforms.ToTensor())
        data_transforms.append(transforms.Normalize(mean, std))
        data_transforms = transforms.Compose(data_transforms)


    def train(self, X, y):
        pass

    def predict(self, X):
        pass


class MLP(NeuralNetClassifier):
    def __init__(self):
        super().__init__()


if __name__== '__main__':
    model = MLP()
    model.transform(0.5, 0.5, RandomResizedCrop=224)

