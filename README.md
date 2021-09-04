# Hybrid Classical-Quantum Convolutional Neural Network for Stenosis Detection in X-ray Coronary Angiography
----------

This repository hosts Python code for Hybrid Classical-Quantum Network (H-CQN). Our paper submitted to *Expert Systems With Applications* on 04 of July 2021. 
----------

<img src="figures/graphical_abstract.png" width="600">

## Abstract
----------
Despite advances in Deep Learning, the Convolutional Neural Networks methods still manifest limitations in medical applications because datasets are usually restricted in the number of samples or include poorly contrasted images. Such a case is found in stenosis detection using X-rays coronary angiography. In this study, the emerging field of quantum computing is applied in the context of hybrid neural networks. So, a hybrid transfer-learning paradigm is used for stenosis detection, where a quantum network
drives and improves the performance of a pre-trained classical network. An intermediate layer between the classical and quantum network post-processes the
classical features by mapping them into a hypersphere of fixed radius through a hyperbolic tangent function. Next, these normalized features are processed in the
quantum network, and through a SoftMax function, the class probabilities are obtained: stenosis and non-stenosis. Furthermore, a distributed variational quantum circuit is
implemented to split the data into multiple quantum circuits within the quantum network, improving the training time without compromising the stenosis detection
performance. The proposed method is evaluated on a small X-ray coronary angiography dataset containing 250 image patches (50%-50% of positive and negative
stenosis cases). The hybrid classical-quantum network significantly outperformed the classical network. Evaluation results showed a boost concerning the classical transfer
learning paradigm in the accuracy of 9%, recall of 20%, and F$_1$-score of 11%, reaching 91.8033%, 94.9153%, and 91.8033%, respectively.

## Dependencies
----------
These are the dependencies to use H-CQN:

* matplotlib (>=3.3.4)
* numpy (>=1.19.5)
* pennylane (>=0.15.1)
* torch (>=1.8.1)
* torchvision (>=0.9.1)
* torchsummary (1.5.1)
* livelossplot (>=0.5.4)

## Quickstart
----------


### Training 
----------

### Testing
----------


## Cite
----------

