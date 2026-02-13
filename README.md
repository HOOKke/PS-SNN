# PS-SNN
## Introduction
This repository contains code supporting the manuscript **“PS-SNN: Pattern Separation Learning for Expandable Spiking Neural Networks in Class-Incremental Learning.”**

![image](https://github.com/HOOKke/PS-SNN/blob/main/images/overview.png)

## Data
The CIFAR-100 dataset analyzed during this study is publicly available at https://www.cs.toronto.edu/~kriz/cifar.html. The ImageNet-100 dataset used in our experiments can be obtained from https://image-net.org/.

## Running experiments

First, You can install the dependencies using the provided requirements.txt file.:
```bash
pip install -r requirements.txt
```


To verify the effectiveness of our method, we conduct experiments on the natural image dataset CIFAR100 under several benchmark protocols. The first protocol, CIFAR100-B0, trains all 100 classes in specific splits over 5 steps (20 classes per step), 10 steps (10 classes per step) and 20 steps (5 classes per step) with a fixed memory size of 2000. The second protocol, CIFAR100-B50, begins with training on 50 classes, after which the remaining 50 classes are introduced incrementally in 5 step (10 classes per step) and 10 step (5 classes per step), with a fixed memory of 20 exemplars per class.

You can see the model code at ./models/spiking_mutable_eb.py

```bash
# CIFAR100-B0-5steps
python main_exp.py --options options/data/cifar100_3orders.yaml options/spiking_eb/spikng_allcomponents_cifar100_B0.yaml \
                --initial-increment 0 --increment 20
                --device <GPU_IDS> --label ps_snn_cifar100_b0_5steps

# CIFAR100-B0-10steps
python main_exp.py --options options/data/cifar100_3orders.yaml options/spiking_eb/spikng_allcomponents_cifar100_B0.yaml \
                --initial-increment 0 --increment 10
                --device <GPU_IDS> --label ps_snn_cifar100_b0_10steps

# CIFAR100-B0-20steps
python main_exp.py --options options/data/cifar100_3orders.yaml options/spiking_eb/spikng_allcomponents_cifar100_B0.yaml \
                --initial-increment 0 --increment 5
                --device <GPU_IDS> --label ps_snn_cifar100_b0_20steps

# CIFAR100-B50-5steps
python main_exp.py --options options/data/cifar100_3orders.yaml options/spiking_eb/spikng_allcomponents_cifar100_B50.yaml \
                --initial-increment 50 --increment 10
                --device <GPU_IDS> --label ps_snn_cifar100_b50_5steps

# CIFAR100-B50-10steps
python main_exp.py --options options/data/cifar100_3orders.yaml options/spiking_eb/spikng_allcomponents_cifar100_B50.yaml \
                --initial-increment 50 --increment 5
                --device <GPU_IDS> --label ps_snn_cifar100_b50_5steps
# or you can also aggregate all the options into one yaml file as options/spiking_eb/spiking_allcomponents_cifar100_inc20b0.yaml


```

## Acknowledgement
Thanks for the fine coding base from [https://github.com/arthurdouillard/incremental_learning.pytorch](https://github.com/arthurdouillard/incremental_learning.pytorch).
