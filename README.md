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

Example Usage

To see more elaborate examples, find some notebooks here.


