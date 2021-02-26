# CVS_framework

## Background

A cognitive vision system (CVS) is any system that uses expert/domain knowledge to model/anticipate visual events. A CVS uses a knowledge representation of a particular visual problem to choose the appropriate strategy, constraints and evaluation criteria for that problem.

From this definition of a CVS, one can imagine a fully comprehensive, general CVS to be similar to that of a human’s visual system. This CVS would carry out a number of complex tasks which would include many aspects of computer vision. For example, object detection and object segmentation might only be minor components of a far more complex system. Furthermore, the way in which these components are used would be wildly varied for different vision tasks (such as pattern recognition, motion perception, etc.). One might even say that the ‘choices’ a general CVS has to make for different vision tasks based on different visual cues/properties (patterns, colours, textures etc.) is subject to combinatorial explosion.

For this reason, it is very difficult to create a general/all encapsulating CVS. Instead, this project aims to build a CVS framework that assists the development of specialised CVSs.

## Aim

The aim of the project is to build a CVS framework that assists developers in creating complex/specialised CVSs.
* A CVS comprises a high-level reasoning component and a low-level image/video processing component. 
* Bayesian Networks (BNs) will be used for the reasoning component of the framework. Its implementation will be specific to computer vision related tasks. The BN implementation can be used to build knowledge representations of problems and/or incorporate domain knowledge into CVSs. 
* The low-level component of the framework will cater to all manner of computer vision related tasks, from image classification and object detection to segmentation.
* An example CVS might achieve scene understanding in video footage. In this case, the low-level component might include the detected/segmented objects and the high-level component would reason about these features to reach some conclusion about the observed scene. 

## Example Usage

To see more examples (with explanatory text), find some notebooks [here](https://github.com/TimothySimons/CVS_framework/tree/master/notebooks).

### Image Classification

```python
from cogvis.classification import nn_classification

train_path = '/path/to/data/train/'
val_path = '/path/to/data/val/'

batch_size = 32
train_transform = nn_classification.data_transform(
    ColorJitter=(),
    RandomHorizontalFlip=(), # only works with PIL images not HDF5
    RandomPerspective=(),    # only works with PIL images not HDf5
    ToTensor=(),
    Normalize=(means, stds), # normalise with custom or calculated means and stds (one mean/std for each channel)
)

val_transform = nn_classification.data_transform(
    ToTensor=(),
    Normalize=(means, stds),
) 

train_size, train_loader = nn_classification.data_loader(train_path, batch_size, 
                                                      transform=train_transform)

val_size, val_loader = nn_classification.data_loader(val_path, batch_size, 
                                                     transform=val_transform)
                                                     
criterion = nn.CrossEntropyLoss()
nn_model, params = nn_classification.existing('resnet18', 10)
optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

nn_model = nn_classification.train(
        nn_model, 
        train_loader, val_loader, 
        train_size, val_size, 
        criterion, 
        optimizer,
        exp_lr_scheduler, 
        num_epochs=10,
        )
        
# ...

preds, probs = nn_classification.predict(nn_model, inputs) # using some provided inputs
```

## Installation

Currently, `cogvis` is not part of the Anaconda distribution or PyPI. You will need to build it from source.

Navigate to the folder containing the `setup.py` project file and execute:

```
python -m pip install --upgrade build
python -m build
```
You will then find a `.whl` file in the newly created `dist` folder.  You can install `cogvis` with a command akin to the following:
```
pip install cogvis-0.0.1-py3-none-any.whl
```
See [here](https://packaging.python.org/tutorials/packaging-projects/) for more information on packaging a Python project.

## Documentation

This Python project uses the [Sphinx autodoc tool](https://www.sphinx-doc.org/en/master/) with RST style doc strings. Take a look at this [Sphinx tutorial](https://sphinx-tutorial.readthedocs.io/) or this [cheat sheet](https://sphinx-tutorial.readthedocs.io/cheatsheet/).

To build documentation, navigate to the docs folder and execute:
```
make html
```
To view the documentation, navigate to the `_build/html` folder and open the `index.html` file in your browser of choice.

> **TIP:**  Try `make clean html` and then rebuild if newly added elements aren't showing.

## Next Steps

Neural Nets:
* instance/semantic segmentation
* object detection

Bayesian Nets:
* create Python interface to common BN C++ library (like dlib)  
_and the following if not provided by the chosen C++ library_
* d-separation
* automatic learning
* plate notation for BN instantiation
* variable elimination (with optimal ordering)

