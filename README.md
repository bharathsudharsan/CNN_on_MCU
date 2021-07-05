# Multi-Component Optimization of Neural-Networks for IoT Devices

Notebook for an end-to-end multi-component NN optimization sequence to enable the execution of high memory plus computation demanding models on MCUs, small CPUs, and AIOT boards. Post multi-component optimization, the resultant models are smaller in size, consume less power when execution, and show low latency. Our sequence is generic and applicable to any state-of-the-art models trained for anomaly detection, predictive maintenance, machine vision, etc.

**Note:** The .ipynb files can be loaded and viewed from the Github page, but it needs to be reloaded a couple of times as the file is big. Hence, it is best to download and open via Google Colab or Jupyter Notebook, thanks.

## Experiment: Multi-component Optimization of CNNs

In the notebooks, we use the standard [MNIST Fashion](https://www.kaggle.com/zalando-research/fashionmnist) (produces CNN1) and [MNIST Digits](http://yann.lecun.com/exdb/mnist/) (produces CNN2) datasets to train a basic CNN whose architecture is shown below

![alt text](https://github.com/bharathsudharsan/CNN_on_MCU/blob/main/Original_CNN_architecture.png)

Both these datasets are imported via the *tf.keras.dataset.name* function with its default train and test sets. After importing, we apply all suitable optimizers before, during, and after training CNNs and analyze the memory conservation, accuracy, and inference speedups. In the following, we breif each optimization component component whose implementaions are provided in the notebooks.

### Pre-training Optimization

We first apply the **pruning** technique on CNNs and present the changes in inference time and size in below Figure c. Similarly, we also perform **quantization-aware training** of CNNs and show the changes in below Figure b. 

![alt text](https://github.com/bharathsudharsan/CNN_on_MCU/blob/main/Pre-training_optimization.png)

### Post-training Optimization

We performed **Int with float fallback quantization** on original CNNs and show its architecture and performance in below Figure d.

![alt text](https://github.com/bharathsudharsan/CNN_on_MCU/blob/main/Int_with_float_quantization_results.png)

Here we quantized the original CNN's Float32 weights and activations to Float16 values. Users can use this **Float16 quantization** when they want to achieve reasonable compression rates (we obtain approx. 6x compression), without loss of precision (we experience only 0.01 % loss in accuracy). Also, Float16 models run on small CPUs without modification. In below Figure e, we show the Float16 quantized model's architecture, inference time, and size changes. 

![alt text](https://github.com/bharathsudharsan/CNN_on_MCU/blob/main/float16_quantization_results.png)

We also converted the CNN's weights & activation to 8-bit integers and show its architecture, inference time, and size changes below Figure f. Here, the size reduced and inference time improved since, after quantization, the inference is carried out using **Integer-only** arithmetic.

![alt text](https://github.com/bharathsudharsan/CNN_on_MCU/blob/main/Int_only_quantization_results.png)


## Results Analysis

We performed analysis based on the experiment results and report the best optimization sequence for:

**Smallest Model Size:** Graph optimized then integer with float fallback quantized version is only 22.5 KB, i.e., 12.06 x times smaller than original CNN. 

**Accuracy Preservation:** Graph optimized then integer only quantized version. For MNIST Fashion, the accuracy increased by 0.27 % and by 0.13 % for MNIST Digits.

**Fast Inference:** Operations optimized then float16 quantized version produces the fastest unit inference results in 0.06 ms.
