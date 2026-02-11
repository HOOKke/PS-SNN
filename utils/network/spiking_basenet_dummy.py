import copy

import torch
from torch import nn

from utils import factory

from spikingjelly.activation_based import layer, functional


class Spiking_BasicNet_dummy(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        init="kaiming",
        device=None,
    ):
        super(Spiking_BasicNet_dummy, self).__init__()
        self.init = init
        self.convnet_type = convnet_type

        #################################################
        # several spiking config
        self.conv_config = cfg.get('conv_config', {})
        self._transfer_conv_config(self.conv_config)
        self.T = cfg['T']
        print(f"The T setting is {self.T}")
        #################################################

        self.convnet = factory.get_convnet(convnet_type, **self.conv_config)
        self.out_dim = self.convnet.out_dim
        self.classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        self.postprocessor = None
        self.to(self.device)
        self.last_classes = 0
        ########################################
        functional.set_step_mode(self, 'm')
        functional.reset_net(self)
        ########################################

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training")

        ########################################
        functional.reset_net(self)
        assert len(x.shape) == 4
        x = x.repeat(self.T, 1, 1, 1, 1)
        ########################################

        features = self.convnet(x)['features']  # -> [T, B, F]
        logits = self.classifier(features)  # -> [T, B, N]

        ########################################
        concat_ft = torch.mean(features, dim=0)  # -> [B, F]
        logits = logits.permute(1, 0, 2)  # -> change to [B, T, N] so that without special problem...
        assert logits.shape[1] == self.T
        ########################################

        return {'feature': concat_ft, 'logit': logits}

    @property
    def features_dim(self):
        return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes).to(self.device)

        del self.classifier
        self.classifier = classifier

        self.last_classes = self.n_classes
        self.n_classes += n_classes

        # SNN NEED TO reset the step method ######################
        functional.set_step_mode(self, 'm')
        ##########################################################

    def _gen_classifier(self, in_features, n_classes):
        classifier = layer.Linear(in_features, n_classes, bias=False)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        # fc = nn.Linear()

        return classifier

    # -> define the spiking parameters
    def _transfer_conv_config(self, conv_config):
        # from spikingjelly.activation_based import neuron
        # neuron.LIFNode()
        conv_config['spiking_neuron'] = factory.get_neuron(conv_config['spiking_neuron'])
        surr = factory.get_surrogate(conv_config['surrogate_function'])
        if surr is None:
            conv_config.pop('surrogate_function')
        else:
            conv_config['surrogate_function'] = surr
