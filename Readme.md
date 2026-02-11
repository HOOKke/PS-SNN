# Adaptive logit align SNN
This repo is a work to explore the method of continual learning in Spiking neural networks (SNN).
The code here mainly explore the dynamic architecture method used in SNNs 

![alade-snn](images/overview.png)

It is also the source code of "ALADE-SNN: Adaptive Logit Alignment in Dynamically Expandable Spiking
Neural Networks for Class Incremental Learning". 
The source paper can be found [here](https://arxiv.org/abs/2412.12696
).

__Note__: We wrote the unit incorrectly in the original paper when comparing ANN's energy consumption with SNN (it's correct when in Openreview). 
It should be "1.76mJ vs 3.67 &mu;J" not "1.76&mu;J vs 3.67pJ". Though it doesn't affect the conclusion.


# Acknowledgement
Thanks for the fine coding base from [https://github.com/arthurdouillard/incremental_learning.pytorch](https://github.com/arthurdouillard/incremental_learning.pytorch).

# Running experiments

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
