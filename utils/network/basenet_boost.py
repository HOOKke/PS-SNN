import copy
import pdb

import torch
from torch import nn
import torch.nn.functional as F

from utils import factory
from utils.network.classifiers import CosineClassifier


class BasicNet_boost(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        use_bias=False,
        init="kaiming",
        device=None,
    ):
        super(BasicNet_boost, self).__init__()
        self.init = init
        self.convnet_type = convnet_type
        self.start_class = cfg['initial_increment']
        self.remove_last_relu = False
        self.use_bias = use_bias
        self.aux_type = cfg['aux_type']  # -> "n+1", "n+n", "n", "none"
        self.der = cfg['der']
        self.conn_fixed = cfg['conn_fixed']
        self.conn_suppress = cfg['conn_suppress']

        self.convnets = nn.ModuleList()
        self.convnets.append(
            factory.get_convnet(convnet_type, remove_last_relu=self.remove_last_relu)
        )
        self.out_dim = self.convnets[0].out_dim

        # 这里需要考虑两个部分的线性分类器是否固定的问题
        self.old_classifier = None
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        raw_features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(raw_features, 1)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) \
            if features.shape[1] > self.out_dim and self.aux_classifier is not None else None

        if self.ntask > 1:
            past_features = torch.cat(raw_features[:-1], 1)
            past_logits = self.old_classifier(past_features)
        else:
            past_logits = None

        return {
            'feature': features, 'past_logit': past_logits,
            'logit': logits, 'aux_logit': aux_logits
        }

    @property
    def features_dim(self):
        return self.out_dim * len(self.convnets)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            # -> copy the parameters of the old extractor #############
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            ###########################################################
            self.convnets.append(new_clf)

        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)

        # if self.classifier is not None and (self.conn_fixed or self.conn_suppress):
        if self.classifier is not None:
            self.old_classifier = copy.deepcopy(self.classifier)
            self.old_classifier.eval()
            for p in self.old_classifier.parameters():
                p.requires_grad = False

        del self.classifier
        self.classifier = fc

        if self.ntask > 1:
            if self.aux_type == "n+1":
                aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
            elif self.aux_type == "n+n":
                aux_fc = self._gen_classifier(self.out_dim, n_classes + self.n_classes)
            elif self.aux_type == "n+m":
                aux_fc = self._gen_classifier(self.out_dim, n_classes + self.ntask - 1)
            elif self.aux_type == 'n':
                aux_fc = self._gen_classifier(self.out_dim, n_classes)
            else:
                aux_fc = None
            del self.aux_classifier
            self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        if self.use_bias:
            nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def copy_convnet(self, idx, src_conv):
        assert idx < len(self.convnets), f"index should be smaller than convnets size, but get {idx}"
        self.convnets[idx].load_state_dict(src_conv.state_dict())

    ##################################################################
    # appended new method for my test
    def weight_ctrl(self):
        if self.ntask <= 1:
            return

        if self.der:
            old_opt, old_ft = self.old_classifier.weight.shape[0], self.old_classifier.weight.shape[1]

            if self.conn_suppress:
                with torch.no_grad():
                    self.classifier.weight.data[old_opt:, :old_ft] = 0.
            if self.conn_fixed:
                self.classifier.weight.data[:old_opt, :old_ft] = self.old_classifier.weight.data

    @property
    def last_dim(self):
        return self.features_dim - self.out_dim
