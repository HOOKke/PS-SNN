import copy
import pdb

import torch
from torch import nn
import torch.nn.functional as F

from utils import factory
from utils.network.classifiers import CosineClassifier


class BasicNet_der(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        use_bias=False,
        init="kaiming",
        device=None,
    ):
        super(BasicNet_der, self).__init__()
        self.init = init
        self.convnet_type = convnet_type
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        self.postprocessor = None

        self.to(self.device)

        ##################################################
        # Appended for a strange test
        self.last_classes = 0
        self.fixed_method = "ignore_o2n"  # "ignore_o2o"  # "all"  # "ignore_o2n"  # "for_itself"  # "ignore_n2o"  # "only_new"
        assert self.fixed_method in ["only_new", "ignore_n2o", "ignore_o2o", "ignore_o2n", "for_itself", "all"]
        print(f"The basenet work use the <{self.fixed_method}> to fix the weight of the classifier")

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            features = [convnet(x)['features'] for convnet in self.convnets]
            # ###############################################
            # # -> stop the gradient calculation so that to reduce the gpu occupation
            # for idx in range(len(features) - 1):
            #     features[idx] = features[idx].detach()
            # ###############################################
            features = torch.cat(features, 1)
        else:
            features = self.convnet(x)['features']

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None

        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
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

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.last_classes = self.n_classes
        print(f"The last classes is {self.last_classes}")
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

        if self.classifier is not None and self.reuse_oldfc:
            weight = copy.deepcopy(self.classifier.weight.data)
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier

    ##################################################################
    # appended new method for my test
    def generate_new_classifier(self):
        fc_grp = nn.Sequential(
            nn.Linear(self.out_dim * len(self.convnets), self.out_dim * len(self.convnets), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim * len(self.convnets), self.n_classes),
        ).to(self.device)

        del self.classifier
        self.classifier = fc_grp

    def fixed_classifier_weight(self):
        with torch.no_grad():
            if self.fixed_method == "all":
                return
            elif self.fixed_method == "only_new":
                self.classifier.weight.data[:, :-self.out_dim] = 0.
            elif self.fixed_method == "ignore_n2o":
                self.classifier.weight.data[:self.last_classes, -self.out_dim:] = 0.
            elif self.fixed_method == "ignore_o2n":
                self.classifier.weight.data[self.last_classes:, :-self.out_dim] = 0.
            elif self.fixed_method == 'ignore_o2o':
                self.classifier.weight.data[:self.last_classes, :-self.out_dim] = 0.
            elif self.fixed_method == 'for_itself':
                self.classifier.weight.data[:self.last_classes, -self.out_dim:] = 0.
                self.classifier.weight.data[self.last_classes:, :-self.out_dim] = 0.
            else:
                raise NotImplementedError(f"{self.fixed_method} is not a valid weight fixation method")

            # weight_norm = torch.norm(self.classifier.weight.data, dim=1)
            # new_norm = torch.mean(weight_norm[self.last_classes:])
            # old_norm = torch.mean(weight_norm[:self.last_classes])
            # w_ctrl = new_norm * 0.75 / old_norm
            # self.classifier.weight.data[:self.last_classes] *= w_ctrl

    def reset_old_clf_weight(self, old_weight):
        assert self.ntask > 1
        with torch.no_grad():
            self.classifier.weight.data[:self.last_classes, :-self.out_dim] = old_weight

    def fix_o2n_weight(self):
        assert self.ntask > 1 and self.last_classes > 0
        with torch.no_grad():
            self.classifier.weight.data[self.last_classes:, :-self.out_dim] = 0.

    def log_cross_clf_weight(self):
        assert self.last_classes > 0
        return self.classifier.weight[self.last_classes-1:self.last_classes+1, -self.out_dim-1:-self.out_dim+1]

    @property
    def last_dim(self):
        return self.features_dim - self.out_dim
