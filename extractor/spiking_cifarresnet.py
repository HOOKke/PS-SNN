import torch
import torch.nn as nn
from torch.utils import model_zoo
from copy import deepcopy
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url
from spikingjelly.activation_based import layer, functional
# -------------------------------------------------- #
#   ResNet Example (Used for cifar dataset in SNN)   #
# -------------------------------------------------- #


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn2(out)

        return out


class Spiking_CifarResNet(nn.Module):
    def __init__(
        self, block, layers, nf=64, zero_init_residual=False, groups=1, width_per_group=64,
        replace_stride_with_dilation=None, norm_layer=None,
        spiking_neuron: callable = None, **kwargs
    ):
        assert spiking_neuron is not None
        super(Spiking_CifarResNet, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = nf
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                  bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))

        self.layer1 = self._make_layer(block, nf * 2, layers[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, nf * 4, layers[1], stride=2, spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, nf * 8, layers[2], stride=2, spiking_neuron=spiking_neuron, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))

        self.out_dim = 8 * nf * block.expansion
        # self.fc1 = layer.Linear(self.out_dim, 100)
        # self.fc_n1 = spiking_neuron(**deepcopy(kwargs))

        # for m in self.modules():
        #     if isinstance(m, layer.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        functional.set_step_mode(self, 'm')
        functional.reset_net(self)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, spiking_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        # functional.reset_net(self)
        # x = x.unsqueeze_(0)
        # x = x.repeat(4, 1, 1, 1, 1)
        # print(x.shape)
        # print(f"neurons membrane at begin: {self.fc_n1.v}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)
        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)

        # x = self.fc1(x)
        # x = self.fc_n1(x)
        # x = x.permute(1, 0, 2)
        # print(f"neurons membrane at end: {self.fc_n1.v.shape}")
        # print(f"output: {x.shape}")
        # return x
        return {"features": x}


def spiking_resnet_cifar(spiking_neuron: callable = None, **kwargs):
    print(f"The spiking neurons' type is {spiking_neuron}")
    if kwargs:
        print(f"And the other kwargs is {kwargs}")
    return Spiking_CifarResNet(BasicBlock, [3, 3, 2], spiking_neuron=spiking_neuron, **kwargs)


if __name__ == '__main__':
    from spikingjelly.activation_based import neuron, surrogate, functional
    from torch.nn import DataParallel
    spk_n = neuron.LIFNode
    surr_func = surrogate.ATan()
    device_idx = [0, 1]
    devices = []
    for idx in device_idx:
        if idx == -1:
            device = torch.device('cpu')
        else:
            device = torch.device(f"cuda:{idx}")
        devices.append(device)

    one_device = devices[0]
    net = spiking_resnet_cifar(spiking_neuron=spk_n, surrogate_function=surr_func).to(devices[0])
    para_net = DataParallel(net, devices)
    functional.set_step_mode(net, 'm')
    functional.reset_net(net)
    T, B = 4, 2

    x = torch.rand(B, 3, 32, 32).to(one_device)
    functional.reset_net(para_net)
    ft_map = para_net(x)['features']
    print(f"after convnet: {ft_map.shape}")


