from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from utils.utils import extract_features, extract_corr_features, extract_logit_features_cls, extract_logit_features


class MemoryBank:

    def __init__(self, device, momentum=0.5):
        self.features = None
        self.targets = None

        self.momentum = momentum

        self.device = device

    def add(self, features, targets):
        if self.features is None:
            self.features = features
            self.targets = targets
        else:
            self.features = torch.cat((self.features, features.to(self.device)), dim=0)
            self.targets = torch.cat((self.targets, targets.to(self.device)), dim=0)

    def get(self, indexes):
        return self.features[indexes]

    def get_neg(self, indexes, n=10):
        neg_indexes = torch.ones(len(self.features)).bool()
        neg_indexes[indexes] = False
        nb = min(n, len(self.features) - len(indexes))
        rnd_indexes = torch.multinomial(torch.ones(nb), nb)

        return self.features[neg_indexes][rnd_indexes]

    def update(self, features, indexes):
        self.features[indexes] = self.momentum * self.features[indexes]\
                                 + (1 - self.momentum * features)


class MemorySetting:
    def __init__(self, mode, total_memory=None, fixed_memory_per_cls=None):
        self.mode = mode
        assert mode.lower() in ["uniform_fixed_per_cls", "uniform_fixed_total_mem", "dynamic_fixed_per_cls"]
        self.total_memory = total_memory
        self.fixed_memory_per_cls = fixed_memory_per_cls
        self._n_classes = 0
        self.mem_per_cls = []

    def update_n_classes(self, n_classes):
        self._n_classes = n_classes

    def update_memory_per_cls_uniform(self, n_classes):
        if "fixed_per_cls" in self.mode:
            self.mem_per_cls = [self.fixed_memory_per_cls for i in range(n_classes)]
        elif "fixed_total_mem" in self.mode:
            self.mem_per_cls = [self.total_memory // n_classes for i in range(n_classes)]
        return self.mem_per_cls

    def update_memory_per_cls(self, network, n_classes, task_size):
        if "uniform" in self.mode:
            self.update_memory_per_cls_uniform(n_classes)
        else:
            if n_classes == task_size:
                self.update_memory_per_cls_uniform(n_classes)

    @property
    def memsize(self):
        if "fixed_total_mem" in self.mode:
            return self.total_memory
        elif "fixed_per_cls" in self.mode:
            return self.fixed_memory_per_cls * self._n_classes


def compute_examplar_mem(feat_norm, feat_flip, herding_mat, nb_max):
    EPSILON = 1e-8
    D = feat_norm.T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)

    D2 = feat_flip.T
    D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

    alph = herding_mat
    alph = (alph > 0) * (alph < nb_max + 1) * 1.0

    alph_mean = alph / np.sum(alph)

    mean = (np.dot(D, alph_mean) + np.dot(D2, alph_mean)) / 2
    # mean = np.dot(D, alph_mean)
    mean /= np.linalg.norm(mean) + EPSILON

    return mean, alph


# -> filtering the features to be store according to the herding strategy recorded in the icarl
def select_examplars(features, nb_max):
    EPSILON = 1e-8
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0], ))
    idxes = []
    w_t = mu

    iter_herding, iter_herding_eff = 0, 0

    while not (np.sum(herding_matrix != 0) == min(nb_max, features.shape[0])) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        # tmp_t = -np.linalg.norm(w_t[:,np.newaxis]-D, axis=0)
        # tmp_t = np.linalg.norm(w_t[:,np.newaxis]-D, axis=0)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            idxes.append(ind_max)
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    return herding_matrix, idxes


# -> randomly filter several data into the memory
def random_selection(n_classes, task_size, network, logger, inc_dataset, memory_per_class: list):
    # TODO: Move data_memroy,targets_memory into IncDataset
    logger.info("Building & updating memory.(Random Selection)")
    tmp_data_memory, tmp_targets_memory = [], []
    assert len(memory_per_class) == n_classes
    for class_idx in range(n_classes):
        if class_idx < n_classes - task_size:
            inputs, targets, loader = inc_dataset.get_custom_loader_from_memory([class_idx])
        else:
            inputs, targets, loader = inc_dataset.get_custom_loader(class_idx, mode="test")
        memory_this_cls = min(memory_per_class[class_idx], inputs.shape[0])
        idxs = np.random.choice(inputs.shape[0], memory_this_cls, replace=False)
        tmp_data_memory.append(inputs[idxs])
        tmp_targets_memory.append(targets[idxs])
    tmp_data_memory = np.concatenate(tmp_data_memory)
    tmp_targets_memory = np.concatenate(tmp_targets_memory)
    return tmp_data_memory, tmp_targets_memory


# -> according the herding rule and generate the key samples' idxes of new classes and return the new memory
def herding(n_classes, task_size, network, herding_matrix, inc_dataset, shared_data_inc, memory_per_class: list,
            logger):
    """Herding matrix: list
    """
    logger.info("Building & updating memory.(iCaRL)")
    tmp_data_memory, tmp_targets_memory = [], []

    for class_idx in range(n_classes):
        inputs = inc_dataset.data_train[inc_dataset.targets_train == class_idx]
        targets = inc_dataset.targets_train[inc_dataset.targets_train == class_idx]
        if class_idx >= n_classes - task_size:  # -> newly coming classes
            # # ImageNet part ########################################
            # if len(shared_data_inc) > len(inc_dataset.targets_inc):
            #     share_memory = [shared_data_inc[i] for i in np.where(inc_dataset.targets_inc == class_idx)[0].tolist()]
            # else:
            #     share_memory = []
            #     for i in np.where(inc_dataset.targets_inc == class_idx)[0].tolist():
            #         if i < len(shared_data_inc):
            #             share_memory.append(shared_data_inc[i])
            # ########################################################

            # share_memory = [shared_data_inc[i] for i in np.where(inc_dataset.targets_inc == class_idx)[0].tolist()]
            loader = inc_dataset._get_loader(
                inputs,
                targets,
                # # ImageNet part ########################################
                # # memory_flags=torch.ones(len(inputs)),
                # # batch_size=128,
                # share_memory=share_memory,
                # ########################################################
                shuffle=False,
                mode="test"  # -> this only concerning about the transform dataset used
            )
            features, _ = extract_features(network, loader)
            # features_flipped, _ = extract_features(network, inc_dataset.get_custom_loader(class_idx, mode="flip")[-1])
            herding_matrix.append(select_examplars(features, memory_per_class[class_idx])[0])
        alph = herding_matrix[class_idx]
        alph = (alph > 0) * (alph < memory_per_class[class_idx] + 1) * 1.0
        # examplar_mean, alph = compute_examplar_mean(features, features_flipped, herding_matrix[class_idx],
        #                                             memory_per_class[class_idx])
        tmp_data_memory.append(inputs[np.where(alph == 1)[0]])
        tmp_targets_memory.append(targets[np.where(alph == 1)[0]])
    tmp_data_memory = np.concatenate(tmp_data_memory)
    tmp_targets_memory = np.concatenate(tmp_targets_memory)
    return tmp_data_memory, tmp_targets_memory, herding_matrix






