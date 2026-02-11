import copy
import pdb

import torch
from torch import nn
import torch.nn.functional as F

from utils import factory


class BasicNet_dummy(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        use_bias=False,
        init="kaiming",
        device=None,
    ):
        super(BasicNet_dummy, self).__init__()
        self.init = init
        self.convnet_type = convnet_type
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = False if not self.weight_normalization else True
        self.use_bias = use_bias if not self.weight_normalization else False

        self.convnet = factory.get_convnet(convnet_type)
        self.out_dim = self.convnet.out_dim
        self.classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        self.postprocessor = None
        self.to(self.device)
        self.last_classes = 0

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training")

        features = self.convnet(x)['features']
        logits = self.classifier(features)

        return {'feature': features, 'logit': logits}

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

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        del self.classifier
        self.classifier = classifier

        self.last_classes = self.n_classes
        self.n_classes += n_classes

    def _gen_classifier(self, in_features, n_classes):
        classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        if self.use_bias:
            nn.init.constant_(classifier.bias, 0.0)

        return classifier

