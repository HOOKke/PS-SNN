import logging
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from.datasets_info import (
    iCIFAR10, iCIFAR100, TinyImageNet200, iImageNet100, iImageNet1000
)
from data_utils.utils import construct_balanced_subset, downsample_subset

logger = logging.getLogger(__name__)


def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == 'cifar10':
        return iCIFAR10
    elif dataset_name == 'cifar100':
        return iCIFAR100
    elif "imagenet100" in dataset_name:
        return iImageNet100
    elif dataset_name == 'imagenet':
        return iImageNet1000
    elif dataset_name == 'tinyimagenet':
        return TinyImageNet200
    else:
        raise NotImplementedError(f"Unknown dataset {dataset_name}")


class IncrementalDataset:
    def __init__(
        self,
        dataset_name,
        random_order=False,  # -> whether to choose class in a random or specific way
        shuffle=True,  # -> shuffle batch order between the epochs
        workers=10,
        batch_size=128,
        seed=1,
        increment=10,  # -> the numbers of classes added each stage
        validation_split=0.,  # -> Percentage of training data to allocate for validation
        # onehot=False,   # -> whether to use a one-hot vector to devote the target rather than a scalar
        initial_increment=None,  # -> whether to train more tasks at the initial steps
        sampler=None,
        sampler_config=None,
        data_path="./store/dataset",
        class_order=None,
        dataset_transforms=None,
        all_test_classes=False,  # -> whether to use ALL the test data
        metadata_path=None,
    ):
        ###########################################
        # Dataset info
        if dataset_transforms is None:
            dataset_transforms = {}
        datasets = _get_datasets(dataset_name)  # -> multi-dataset setting
        if metadata_path:  # -> it seems like a special setting for ImageNet
            print(f"Adding metadata path {metadata_path}")
            datasets[0].metadata_path = metadata_path
        # -> get the whole self.data_train and other data...
        self._setup_data(
            datasets,
            random_order=random_order,
            class_order=class_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split,
            initial_increment=initial_increment,
            data_path=data_path,
        )

        dataset = datasets[0]()
        # -> the custom transform
        dataset.set_custom_transforms(dataset_transforms)
        self.train_transforms = dataset.train_transforms
        self.test_transforms = dataset.test_transforms
        self.common_transforms = dataset.common_transforms

        self.open_image = datasets[0].open_image

        self._seed = seed
        self._workers = workers
        self._shuffle = shuffle
        self._batch_size = batch_size
        # self._onehot = onehot
        self._sampler = sampler
        self._sampler_config = sampler_config
        self._all_test_classes = all_test_classes

        ###########################################
        # Memory Setting
        self.data_memory = None  # -> memory data of inputs
        self.targets_memory = None
        # Incoming data D_t
        self.data_cur = None
        self.targets_cur = None
        # All available data current stage, including memory and the cur ingoing data
        self.data_inc = None
        self.targets_inc = None
        # -> possibly meaningless
        self.data_test_inc = None
        self.targets_test_inc = None

        self._current_task = 0

        # # ImageNet part ########################################
        # # -> I guess it is because the ImageNet only store the complete picture
        # self.shared_data_inc = None
        # self.shared_test_data = None
        # ########################################################

    @property
    def n_tasks(self):
        return len(self.increments)

    @property
    def n_classes(self):
        return sum(self.increments)

    def new_task(self):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks")

        min_class, max_class, x_train, y_train, x_val, y_val, x_test, y_test = \
            self._get_cur_step_data_for_raw_data()
        nb_new_classes = len(np.unique(y_train))
        # if self._onehot:
        #     def to_onehot(x):
        #         n = np.max(x) + 1
        #         return np.eye(n)[x]
        #
        #     y_train = to_onehot(y_train)

        self.data_cur, self.targets_cur = x_train, y_train

        if self.data_memory is not None:
            logger.info(f"Set memory of size: {len(self.data_memory)}")
            if len(self.data_memory) != 0:
                x_train = np.concatenate((x_train, self.data_memory))
                y_train = np.concatenate((y_train, self.targets_memory))
        else:
            logger.info("MEMORY IS NONE!")

        self.data_inc, self.targets_inc = x_train, y_train
        self.data_test_inc, self.targets_test_inc = x_test, y_test

        train_loader = self._get_loader(x_train, y_train, mode="train")
        val_loader = self._get_loader(x_val, y_val, shuffle=False, mode="test") if len(x_val) > 0 else None
        test_loader = self._get_loader(x_test, y_test, mode="test")

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "total_n_classes": sum(self.increments),
            "increment": nb_new_classes,
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": x_train.shape[0],
            "n_test_data": x_test.shape[0],
        }
        self._current_task += 1
        return task_info, train_loader, val_loader, test_loader

    def _get_cur_step_data_for_raw_data(self):
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])

        x_train, y_train = self._select(self.data_train, self.targets_train, min_idx=min_class, max_idx=max_class)
        x_val, y_val = self._select(self.data_val, self.targets_val, min_idx=min_class, max_idx=max_class)
        if self._all_test_classes is True:
            logger.info("Testing on all classes!")
            test_max = sum(self.increments)
        elif self._all_test_classes is not None and self._all_test_classes is not False:
            test_max = sum(self.increments[:min(self._current_task+1+self._all_test_classes, len(self.increments))])
            logger.info(
                f"Test on {self._all_test_classes} unseen tasks (max class = {max_class})"
            )
        else:
            logger.info(f"Test on All classes up to learning periods (max class = {max_class})")
            test_max = max_class

        x_test, y_test = self._select(self.data_test, self.targets_test, max_idx=test_max)
        return min_class, max_class, x_train, y_train, x_val, y_val, x_test, y_test

    def _add_memory(self, x, y, data_memory, targets_memory):
        # if self._onehot:  # -> to append the extra zero
        #     targets_memory = np.concatenate(
        #         (
        #             targets_memory,
        #             np.zeros((targets_memory.shape[0], self.increments[self._current_task]))
        #         ), axis=1
        #     )
        # -> so the memory_flags is used to mark the sampler item and the task item...
        memory_flags = np.concatenate((np.zeros((x.shape[0], )), np.ones((data_memory.shape[0], ))))
        x = np.concatenate((x, data_memory))
        y = np.concatenate((y, targets_memory))

        return x, y, memory_flags

    def get_custom_loader_from_memory(self, class_indexes, mode="test"):
        if not isinstance(class_indexes, list):
            class_indexes = [class_indexes]
        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(self.data_memory,
                                                     self.targets_memory,
                                                     min_idx=class_index,
                                                     max_idx=class_index + 1)
            data.append(class_data)
            targets.append(class_targets)

        data = np.concatenate(data)
        targets = np.concatenate(targets)
        shuffle = False if mode == "test" else True

        return data, targets, self._get_loader(data, targets, shuffle=shuffle, mode=mode)

    def get_custom_loader(
        self, class_indexes, mode="test", data_source="train", sampler=None
    ):
        """Returns a custom loader.

        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if not isinstance(class_indexes, list):
            class_indexes = [class_indexes]

        if data_source == "train":
            # x, y = self.data_inc, self.targets_inc
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            if len(self.data_val) > 0:
                x, y = self.data_val, self.targets_val
            else:
                x, y = self.data_test, self.targets_test
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError(f"Unknown data source <{data_source}>")

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(
                x, y, min_idx=class_index, max_idx=class_index+1
            )
            data.append(class_data)
            targets.append(class_targets)

        data = np.concatenate(data)
        targets = np.concatenate(targets)

        shuffle = True if "train" in mode else False
        return data, targets, self._get_loader(
            data, targets, shuffle=shuffle, mode=mode, sampler=sampler
        )

    def _get_memory_loader(self, data, targets):
        return self._get_loader(
            data, targets, shuffle=True, mode="train"
        )

    def _select(self, x, y, min_idx=0, max_idx=0):
        assert min_idx < max_idx
        # idxes = np.where(np.logical_and(y >= min_idx, y < max_idx))[0]
        # return x[idxes], y[idxes]
        idxes = sorted(np.where(np.logical_and(y >= min_idx, y < max_idx))[0])
        if isinstance(x, list):
            selected_x = [x[idx] for idx in idxes]
        else:
            selected_x = x[idxes]
        return selected_x, y[idxes]

    # # ImageNet part ########################################
    # def _get_loader(self, x, y, share_memory=None, shuffle=True, mode="train", sampler=None, use_sampler=True):
    # ########################################################
    def _get_loader(self, x, y, shuffle=True, mode="train", sampler=None, use_sampler=True):
        if "balanced" in mode:
            x, y = construct_balanced_subset(x, y)
            use_sampler = False
        if "downsample" in mode:
            from utils.const import DOWNSAMPLE_SIZE
            x, y = downsample_subset(x, y, DOWNSAMPLE_SIZE)
            use_sampler = False

        batch_size = self._batch_size

        if "train" in mode:
            trsf = transforms.Compose([*self.train_transforms, *self.common_transforms])
            sampler = self._sampler if use_sampler else None
            if sampler is not None and self._current_task > 0:
                logger.info(f"Using sampler {sampler}")
                memory_flags = torch.zeros(y.shape[0])
                memory_flags[y < sum(self.increments[:self._current_task])] = 1
                sampler = sampler(y, memory_flags, batch_size=self._batch_size, **self._sampler_config)
                batch_size = 1
            else:
                sampler = None
        elif "test" in mode:
            trsf = transforms.Compose([*self.test_transforms, *self.common_transforms])
            sampler = None
        elif "flip" in mode:
            trsf = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=1.), *self.test_transforms,
                    *self.common_transforms
                ]
            )
            sampler = None
        else:
            raise NotImplementedError(f"Unknown mode {mode}.")

        return DataLoader(
            DummyDataset(x, y, trsf, open_image=self.open_image),
            # # ImageNet part ########################################
            # DummyDataset(x, y, trsf, share_memory_=share_memory, open_image=self.open_image),
            # ########################################################
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=self._workers,
            batch_sampler=sampler,
        )

    def _setup_data(
        self,
        datasets,
        random_order=False,
        class_order=None,
        seed=1,
        increment=10,
        validation_split=0.,
        initial_increment=None,
        data_path="./store/dataset"
    ):
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.increments = []
        self.class_order = []

        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:
            train_dataset = dataset().base_dataset(data_path, train=True, download=True)
            test_dataset = dataset().base_dataset(data_path, train=False, download=True)

            x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
            x_val, y_val, x_train, y_train = self._split_per_class(
                x_train, y_train, validation_split
            )
            x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

            order = list(range(len(np.unique(y_train))))
            if random_order:
                random.seed(seed)
                random.shuffle(order)
            elif class_order:
                order = class_order
            elif dataset.class_order is not None:
                order = dataset.class_order
            elif hasattr(train_dataset, "class_order") and train_dataset.class_order is not None:
                order = train_dataset.class_order

            logger.info(f"Dataset {dataset.__name__}: class ordering: {order}.")
            self.class_order.append(order)
            y_train = self._map_new_class_index(y_train, order)
            y_val = self._map_new_class_index(y_val, order)
            y_test = self._map_new_class_index(y_test, order)

            # -> add the classes' ids that already be taken into account
            y_train += current_class_idx
            y_val += current_class_idx
            y_test += current_class_idx

            current_class_idx += len(order)
            if len(datasets) > 1:  # -> the condition that incremental learning between different distributions
                self.increments.append(len(order))
            else:
                self._split_increment(len(order), increment=increment, initial_increment=initial_increment)

            self.data_train.append(x_train)
            self.targets_train.append(y_train)
            self.data_val.append(x_val)
            self.targets_val.append(y_val)
            self.data_test.append(x_test)
            self.targets_test.append(y_test)

        # -> IT IS CONFUSING TO DIRECTLY JOIN THE MULTIPLE TRAINING DATA TOGETHER
        self.data_train = np.concatenate(self.data_train)
        self.targets_train = np.concatenate(self.targets_train)
        self.data_val = np.concatenate(self.data_val)
        self.targets_val = np.concatenate(self.targets_val)
        self.data_test = np.concatenate(self.data_test)
        self.targets_test = np.concatenate(self.targets_test)

    def _split_increment(self, n_classes, increment, initial_increment=None):
        if initial_increment is None or initial_increment == 0:
            inc_steps = n_classes / increment
            remainder = n_classes - int(inc_steps) * increment  # -> the remaining classes?
            # -> the remainder will be added in the last
            if not inc_steps.is_integer():
                logger.warning(
                    f"THe last step will have slightly less sample ({remainder} vs {increment})."
                )
                self.increments = [increment for _ in range(int(inc_steps))]
                self.increments.append(remainder)
            else:
                self.increments = [increment for _ in range(int(inc_steps))]

        else:
            self.increments = [initial_increment]

            inc_steps = (n_classes - initial_increment) / increment
            remainder = (n_classes - initial_increment) - int(inc_steps) * increment
            if not inc_steps.is_integer():
                logger.warning(
                    f"The last step will have slightly less sample ({remainder} vs {increment})."
                )
                self.increments.extend([increment for _ in range(int(inc_steps))])
                self.increments.append(remainder)
            else:
                self.increments.extend([increment for _ in range(int(inc_steps))])

    @staticmethod  # -> modify labels according to experimental order
    def _map_new_class_index(y, order):
        return np.array(list(map(lambda x: order.index(x), y)))

    @staticmethod  # -> split a validation dataset from the training dataset
    def _split_per_class(x, y, validation_split=0.):
        # -> shuffle the indexes of the datasets
        shuffled_indexes = np.random.permutation(x.shape[0])
        x = x[shuffled_indexes]
        y = y[shuffled_indexes]

        x_val, y_val = [], []
        x_train, y_train = [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_num = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_num]
            train_indexes = class_indexes[nb_val_num:]

            x_val.append(x[val_indexes])
            y_val.append(y[val_indexes])
            x_train.append(x[train_indexes])
            y_train.append(y[train_indexes])

        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        return x_val, y_val, x_train, y_train

    def combine_two_loader(self, train_loader, val_loader):
        train_set = train_loader.dataset
        val_set = val_loader.dataset
        # concat_set = ConcatDataset([train_loader.dataset, val_loader.dataset])
        new_inputs = np.concatenate((train_set.x, val_set.x))
        new_lbls = np.concatenate((train_set.y, val_set.y))

        return self._get_loader(new_inputs, new_lbls, mode="train")


# -> DER use a shared memory to process with the ImageNet... I do not need this for now
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, trsf, open_image=False):
        self.x, self.y = x, y
        self.trsf = trsf  # -> the preset transformation for the dataset
        self.open_image = open_image

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        if self.open_image:
            img = Image.open(x).convert("RGB")
        else:
            img = Image.fromarray(x.astype('uint8'))

        img = self.trsf(img)
        return img, y


# # ImageNet part ########################################
# import multiprocessing as mp
# class DummyDataset(torch.utils.data.Dataset):
#     def __init__(self, x, y, trsf, share_memory_=None, open_image=False):
#         self.x, self.y = x, y
#         self.trsf = trsf  # -> the preset transformation for the dataset
#         self.open_image = open_image
#         self.manager = mp.Manager()
#         # self.buffer_size = 4000000
#         self.buffer_size = 2000000
#         if share_memory_ is None:
#             if self.x.shape[0] > self.buffer_size:
#                 self.share_memory = self.manager.list([None for _ in range(self.buffer_size)])
#             else:
#                 self.share_memory = self.manager.list([None for _ in range(len(x))])
#         else:
#             self.share_memory = share_memory_
#
#     def __len__(self):
#         return self.x.shape[0]
#
#     def __getitem__(self, idx):
#         x, y = self.x[idx], self.y[idx]
#
#         if self.open_image:
#             if idx < len(self.share_memory):
#                 if self.share_memory[idx] is not None:
#                     img = self.share_memory[idx]
#                 else:
#                     img = Image.open(x).convert("RGB")
#                     self.share_memory[idx] = img
#             else:
#                 img = Image.open(x).convert("RGB")
#         else:
#             img = Image.fromarray(x.astype('uint8'))
#
#         img = self.trsf(img)
#         return img, y
# ########################################################


