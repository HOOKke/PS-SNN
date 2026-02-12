# PS-SNN
## Introduction
This repository contains code supporting the manuscript **“PS-SNN: Pattern Separation Learning for Expandable Spiking Neural Networks in Class-Incremental Learning.”**

![image](https://github.com/HOOKke/PS-SNN/tree/main/images)

## Data
Our experiments relies on the natural image dataset CIFAR100(https://www.cs.toronto.edu/~kriz/cifar.html).

## Running experiments

First install the required package:
```bash
pip install -r requirements.txt
```

The code here main deploy this method on CIFAR100. With two benchmarks, including CIFAR100-B0-inc10 and CIFAR100-B50-inc10 and so on.

You can see the model code at ./models/spiking_mutable_eb.py

```bash
# CIFAR100-B0-5steps
python main_exp.py --options options/data/cifar100_3orders.yaml options/spiking_eb/spikng_allcomponents_cifar100_B0.yaml \
                --initial-increment 0 --increment 20
                --device <GPU_IDS> --label alade_snn_cifar100_b0_5steps

# CIFAR100-B0-10steps
python main_exp.py --options options/data/cifar100_3orders.yaml options/spiking_eb/spikng_allcomponents_cifar100_B0.yaml \
                --initial-increment 0 --increment 10
                --device <GPU_IDS> --label alade_snn_cifar100_b0_10steps

# CIFAR100-B50-5steps
python main_exp.py --options options/data/cifar100_3orders.yaml options/spiking_eb/spikng_allcomponents_cifar100_B50.yaml \
                --initial-increment 50 --increment 10
                --device <GPU_IDS> --label alade_snn_cifar100_b50_5steps
# or you can also aggregate all the options into one yaml file as options/spiking_eb/spiking_allcomponents_cifar100_inc20b0.yaml

# DVS-CIFAR100-B0-2steps
python main_exp.py --options options/spiking_eb/spikng_allcomponents_dvscifar10_inc5b0.yaml \
                --device <GPU_IDS>
  
# The options below has not been best fine-tuned though, a large time-window shall perform better
# TINYIMAGENET-B0-10steps
python main_exp.py --options options/spiking_eb/spikng_allcomponents_tinyImageNet_B0.yaml \
                --device <GPU_IDS>

python main_exp.py --options options/data/imagenet100_1order.yaml options/spiking_eb/spikng_allcomponents_tinyImageNet_B0.yaml \
                --initial-increment 0 --increment 10
                --device <GPU_IDS> --label alade_snn_imagenet100_b0_10steps

```

## Acknowledgement
Thanks for the fine coding base from [https://github.com/arthurdouillard/incremental_learning.pytorch](https://github.com/arthurdouillard/incremental_learning.pytorch).
