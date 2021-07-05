# Multi-Component Optimization of Neural-Networks for IoT Devices

Notebook for an end-to-end multi-component NN optimization sequence to enable the execution of high memory plus computation demanding models on MCUs, small CPUs, and AIOT boards. Post multi-component optimization, the resultant models are smaller in size, consume less power when execution, and show low latency. Our sequence is generic and applicable to any state-of-the-art models trained for anomaly detection, predictive maintenance, machine vision, etc.

In the notebooks, we use the standard [MNIST Fashion](https://www.kaggle.com/zalando-research/fashionmnist) (produces CNN1) and [MNIST Digits](http://yann.lecun.com/exdb/mnist/) (produces CNN2) datasets to train a basic CNN whose architecture is shown below

![alt text] (https://github.com/bharathsudharsan/CNN_on_MCU/blob/main/Original_CNN_architecture.png)

Both these datasets are imported via the *tf.keras.dataset.name* function with its default train and test sets. After importing, we apply all suitable optimizers before, during, and after training CNNs and analyze the memory conservation, accuracy, and inference speedups.


**Note:** The .ipynb files can be loaded and viewed from the Github page, but it needs to be reloaded a couple times as the file is big. Hence, it is best to download and open via Google Colab or Jupyter Notebook, thanks.




