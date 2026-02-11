import copy
import pdb

import torch
from torch import nn
import torch.nn.functional as F

from utils import factory

from spikingjelly.activation_based import layer, functional
from copy import deepcopy


class Spiking_BasicNet_eb(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        init="kaiming",
        device=None,
    ):
        super(Spiking_BasicNet_eb, self).__init__()
        self.cfg = cfg
        self.init = init
        self.convnet_type = convnet_type

        # self.spiking = cfg['spiking'] -> I decide to put this on the upper level
        #################################################
        # several spiking config
        self.conv_config = deepcopy(cfg.get('conv_config', {}))
        self._transfer_conv_config(self.conv_config)
        self.T = cfg['T']
        print(self.conv_config['spiking_neuron'])
        print(f"self.T: {self.T}")
        #################################################

        self.aux_nplus1 = cfg['aux_n+1']
        self.convnets = nn.ModuleList()
        self.convnets.append(
            factory.get_convnet(convnet_type, **self.conv_config)
        )
        self.out_dim = self.convnets[0].out_dim

        self.classifier = None
        self.aux_clf = None

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


        self.total_classes = len(self.cfg["class_order"][0])
        self.initial_increment = self.cfg['initial_increment']
        self.increment = self.cfg['increment']
        
        if self.cfg.get("orthogonal_init_clf", False):
            self.orthogonal_init_clf = True
        else:
            self.orthogonal_init_clf = False
        
        if self.cfg.get("fix_clf", False):
            self.fix_clf = True
        else:
            self.fix_clf = False

        # 初始化正交分类器权重
        self.orthogonal_clf_weight = None
        if self.orthogonal_init_clf:
            # 生成 out_dim * total_classes 的正交矩阵
            self.orthogonal_clf_weight = self._generate_orthogonal_weight(self.out_dim, self.total_classes)

    def forward(self, x: torch.Tensor):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        ########################################
        functional.reset_net(self)
        # assert len(x.shape) == 4
        # x = x.repeat(self.T, 1, 1, 1, 1)
        # -> Considering the condition when processing with DVS-datasets
        if len(x.shape) == 4:
            x = x.repeat(self.T, 1, 1, 1, 1)
        else:
            assert x.shape[1] == self.T
            x = x.permute(1, 0, 2, 3, 4)  # -> [B, T, C, H, W] -> [T, B, C, H, W]
        ########################################

        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, dim=-1)  # -> [T, B, F]
        logits = self.classifier(features)  # -> [T, B, N]

        ########################################
        concat_ft = torch.mean(features, dim=0)  # -> [B, F]
        logits = logits.permute(1, 0, 2)  # -> [B, T, N]
        assert logits.shape[1] == self.T
        ########################################

        if self.aux_clf is not None:
            aux_logits = self.aux_clf(features[..., -self.out_dim:])  # -> [T, B, N]
            ########################################
            aux_logits = aux_logits.permute(1, 0, 2)  # -> [B, T, N]
            ########################################
        else:
            aux_logits = None

        return {'feature': concat_ft, 'logit': logits, 'aux_logit': aux_logits}

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

        self._add_classes_expanding(n_classes)
        self.last_classes = self.n_classes
        self.n_classes += n_classes

        # SNN NEED TO reset the step method ######################
        functional.set_step_mode(self, 'm')
        ##########################################################

    def _add_classes_expanding(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type, **self.conv_config).to(self.device)
            # -> copy the parameters of the old extractor #############
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            ###########################################################
            self.convnets.append(new_clf)

        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes, may_orth_work=True)
        del self.classifier
        self.classifier = fc

        if self.ntask > 1:
            if self.aux_nplus1:
                aux_fc = self._gen_classifier(self.out_dim, 1 + n_classes, may_orth_work=False)
            else:
                aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
            del self.aux_clf
            self.aux_clf = aux_fc

    def _gen_classifier(self, in_features, n_classes, may_orth_work=False):
        classifier = layer.Linear(in_features, n_classes, bias=False).to(self.device)

        if may_orth_work and self.orthogonal_clf_weight is not None:
            # 复制正交权重
            # self.orthogonal_clf_weight: [out_dim, total_classes]
            # 复制len(self.convnets)倍 -> [out_dim * len(self.convnets), total_classes]
            repeat_num = in_features // self.out_dim
            weight = self.orthogonal_clf_weight.repeat(repeat_num, 1)  # [out_dim * len(self.convnets), total_classes]
            # 取前n_classes列
            weight = weight[:, :n_classes]
            with torch.no_grad():
                classifier.weight.data.copy_(weight.T)  # Linear的weight shape: [n_classes, in_features]
            if self.fix_clf:
                classifier.weight.requires_grad = False
        else:
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")

        return classifier

    def _generate_orthogonal_weight(self, out_dim, total_classes):
        # 生成正交矩阵，shape: [out_dim, total_classes]
        # 若total_classes > out_dim，先生成total_classes * total_classes的正交矩阵再取前out_dim行
        # 若out_dim > total_classes，先生成out_dim * out_dim的正交矩阵再取前total_classes列
        if total_classes <= out_dim:
            # 生成out_dim * out_dim正交矩阵，取前total_classes列
            q, _ = torch.linalg.qr(torch.randn(out_dim, out_dim))
            return q[:, :total_classes].to(self.device)
        else:
            # 生成total_classes * total_classes正交矩阵，取前out_dim行
            q, _ = torch.linalg.qr(torch.randn(total_classes, total_classes))
            return q[:out_dim, :].to(self.device)
    # -> define the spiking parameters
    def _transfer_conv_config(self, conv_config):
        conv_config['spiking_neuron'] = factory.get_neuron(conv_config['spiking_neuron'])
        surr = factory.get_surrogate(conv_config['surrogate_function'])
        if surr is None:
            conv_config.pop('surrogate_function')
        else:
            conv_config['surrogate_function'] = surr
    #############################################
    # Append method for some special operation...
    #############################################
    def reset_old_clf_weight(self, old_weight):
        assert self.ntask > 1
        with torch.no_grad():
            self.classifier.weight.data[:self.last_classes, :-self.out_dim] = old_weight
            self.classifier.weight.data[self.last_classes:, :-self.out_dim] = 0.

    def parameter_constraint(self, constr_type):
        assert self.ntask > 1 and self.last_classes > 0
        with torch.no_grad():
            if constr_type == "o2n":
                self.classifier.weight.data[self.last_classes:, :-self.out_dim] = 0.
            elif constr_type == "only_new":
                self.classifier.weight.data[:, :-self.out_dim] = 0.
            elif constr_type == 'none':
                pass
            else:
                raise NotImplementedError(f"{constr_type} is not a valid weight fixation")

    @property
    def last_dim(self):
        return self.features_dim - self.out_dim

    @property
    def clf_weight(self):
        return self.classifier.weight.data

