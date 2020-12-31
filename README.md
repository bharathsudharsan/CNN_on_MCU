# Multi-Component Optimization and Efficient Deployment of Neural-Networks on Resource-Constrained IoT Devices

**Overview:** The repo contains the code for an end-to-end multi-component NN optimizer to enable the execution of high memory and computation demanding models on MCUs, small CPUs, and AIOT boards. When any Machine Learning (ML) model is optimization sequence, the resultant models are much smaller in size, consume less power when execution, and show low latency. Our sequence is generic and applicable to any state-of-the-art models trained for anomaly detection, predictivemaintenance, machine vision, etc.

**Datasets:** For the experiments, we use the standard [MNIST Fashion](https://www.kaggle.com/zalando-research/fashionmnist) (produces CNN1) and [MNIST Digits](http://yann.lecun.com/exdb/mnist/) (produces CNN2) datasets to train a basic CNN. 

In the remainder, we brief each optimizer component of the proposed end-to-end multi-component optimization sequence.

**Pre-training Optimization:** We provide implementations of two popular pre-training optimizers that can optimize the video analytics algorithm to become more resource friendly. Pruning and Quantization-aware Training is performed on the CNNs.

**Post-training Optimization:** We implement and provide methods that quantize the models by reducing the precision of their weights to save memory and simplify calculations often without much impact on accuracy. 

**Operations Optimization:** When designing models to execute on low-resource devices, only limited operations can be used to keep the operational cost low. Here, we implement and provide an operations optimization technique that is a part of our sequence.

**Graph Optimization:** The graph of an ML model contains nodes and edges. We provide an implementation of Graph Optimizers that users can leverage to optimize the model graphs to improve the computational performance while reducing peak SRAM (memory) usage on MCUs, thus enabling the execution of larger models on tiny memory footprints.

In our paper, we apply all suitable components of our optimization sequence before, during, and after the training of CNNs, and report the memory conservation and inference speedups. We additionally present joint model optimization methods and approaches to reduce model workload and improve kernel performance.


