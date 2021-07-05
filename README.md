# Multi-Component Optimization of Neural-Networks for IoT Devices

Notebook for an end-to-end multi-component NN optimization sequence to enable the execution of high memory plus computation demanding models on MCUs, small CPUs, and AIOT boards. Post multi-component optimization, the resultant models are smaller in size, consume less power when execution, and show low latency. Our sequence is generic and applicable to any state-of-the-art models trained for anomaly detection, predictive maintenance, machine vision, etc.

## Experiment: Multi-component Optimization of CNNs

In the notebooks, we use the standard [MNIST Fashion](https://www.kaggle.com/zalando-research/fashionmnist) (produces CNN1) and [MNIST Digits](http://yann.lecun.com/exdb/mnist/) (produces CNN2) datasets to train a basic CNN whose architecture is shown below

![alt text](https://github.com/bharathsudharsan/CNN_on_MCU/blob/main/Original_CNN_architecture.png)

Both these datasets are imported via the *tf.keras.dataset.name* function with its default train and test sets. After importing, we apply all suitable optimizers before, during, and after training CNNs and analyze the memory conservation, accuracy, and inference speedups.

**Note:** The .ipynb files can be loaded and viewed from the Github page, but it needs to be reloaded a couple times as the file is big. Hence, it is best to download and open via Google Colab or Jupyter Notebook, thanks.

## Results Analysis

We performed analysis based on the experiment results:

**Best Optimization Sequence for Smallest Model Size:** Graph optimized then integer with float fallback quantized version is only 22.5 KB, and 12.06 x times smaller than original CNN. 

**Best Optimization Sequence for Accuracy Preservation:** Graph optimized then integer only quantized version. For MNIST Fashion, the accuracy increased by 0.27 %, and by 0.13 % for MNIST Digits.

**Best Optimization Sequence for Fast Inference:** Operations optimized then float16 quantized version produces the fastest unit inference results in 0.06 ms.





