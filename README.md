# Hybrid Classical-Quantum Convolutional Neural Network for Stenosis Detection in X-ray Coronary Angiography
----------

This repository hosts Python code for the paper **Hybrid Classical-Quantum Convolutional Neural Network for Stenosis Detection in X-ray Coronary Angiography**, published at [Expert Systems With Applications](https://doi.org/10.1016/j.eswa.2021.116112) on 28 October 2021. 

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
If you use this for research, please cite. Here is an example BibTeX entry:

```
@article{ovalle2021hybrid,
title = {{Hybrid classical-quantum convolutional neural network for stenosis detection in X-ray coronary angiography}},
author = {Emmanuel Ovalle-Magallanes and Juan Gabriel Avina-Cervantes and Ivan Cruz-Aceves and Jose Ruiz-Pinales},
journal = {Expert Systems with Applications},
pages = {116112},
year = {2021},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2021.116112},
}
```
----------
## Development

Want to contribute? Great!. Contact us.

Follow our research work at: 
* [Personal Page](https://emmanuelovalle.netlify.app/)
* [Researchgate](https://www.researchgate.net/profile/Emmanuel-Ovalle-Magallanes)
* [Google Academic](https://scholar.google.com/citations?user=zql1lk8AAAAJ&hl=es#)

