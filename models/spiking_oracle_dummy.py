import numpy as np
import random
import time
import math
import logging
import os
from copy import deepcopy
from scipy.spatial.distance import cdist

import torch
from torch.nn import DataParallel
from torch.nn import functional as F

from utils import network, factory, utils
from models import IncrementalLearner
from utils.metrics import ClassErrorMeter
from utils.schedulers import GradualWarmupScheduler

# Constants
EPSILON = 1e-8
logger = logging.getLogger(__name__)


class Spiking_Oracle_Dummy(IncrementalLearner):
    def __init__(self, cfg, inc_dataset=None, tensorboard=None, ):
        super().__init__()
        self._cfg = cfg
        self._device = cfg['device'][0]
        self._multi_devices = cfg['device']

        ###########################################
        # Network settings
        self._network = network.Spiking_BasicNet_dummy(
            cfg["convnet"], cfg=cfg,
            device=self._device
        )
        self._parallel_network = DataParallel(self._network, self._multi_devices)

        ###########################################
        # Data
        self._inc_dataset = inc_dataset
        self._n_classes = 0

        ###########################################
        # Logging
        assert tensorboard is not None
        self._tensorboard = tensorboard
        self._val_per_n_epoch = cfg["val_per_n_epoch"]

        if cfg.get("exp_dir", None) is None:
            ckpt_dir = os.path.join(cfg['opt_dir'], 'default', 'ckpts')
        else:
            ckpt_dir = os.path.join(cfg['exp_dir'], 'ckpts')
        self.save_path = ckpt_dir
        self._save_ckpt = cfg["save_ckpt"]
        if self._save_ckpt:
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)

        ###########################################
        # Training
        self._n_epochs = cfg["epochs"]
        self._lr = cfg["lr"]
        self._lr_decay = cfg["lr_decay"]
        self._opt_name = cfg["optimizer"]
        self._scheduling = cfg["scheduling"]
        self._warmup = cfg['warmup']
        self._weight_decay = cfg["weight_decay"]
        self._init_config = cfg.get("initial_training_config", None)
        if self._init_config is not None:
            assert cfg["initial_increment"] >= 0
        else:
            self._init_config = {}

        self._optimizer = None
        self._scheduler = None

        ###########################################
        # Network special settings
        self.TET_config = cfg['TET_config']
        self._analyse = cfg['analyse']

    def eval(self):
        self._parallel_network.eval()

    def train(self):
        self._parallel_network.train()

    def _before_task(self, taski, inc_dataset):
        self._task = taski
        self._n_classes += self._task_size

        if taski == 0 and self._init_config != {}:
            logger.info(f"Take special setting for the first trail: {self._init_config}")
            self._n_epochs = self._init_config['epochs']
            self._lr = self._init_config['lr']
            self._lr_decay = self._init_config['lr_decay']
            self._opt_name = self._init_config['optimizer']
            self._scheduling = self._init_config['scheduling']
            self._weight_decay = self._init_config['weight_decay']
        else:
            self._n_epochs = self._cfg["epochs"]
            self._lr = self._cfg["lr"]
            self._lr_decay = self._cfg["lr_decay"]
            self._opt_name = self._cfg["optimizer"]
            self._scheduling = self._cfg["scheduling"]
            self._weight_decay = self._cfg["weight_decay"]

        self._network.add_classes(self._task_size)
        # self._network.task_size = self._task_size
        self.set_optimizer()

    def set_optimizer(self, lr=None):
        if lr is None:
            lr = self._lr

        if self._cfg["dynamic_weight_decay"]:
            # used in BiC official implementation
            weight_decay = self._weight_decay * self._cfg["task_max"] / (self._task + 1)
        else:
            weight_decay = self._weight_decay
        logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))

        self._optimizer = factory.get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()),
                                                self._opt_name, lr, weight_decay)

        if "cos" in self._cfg["scheduler"]:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self._n_epochs)
        else:
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self._optimizer, self._scheduling, gamma=self._lr_decay
            )

        if self._warmup:
            print("warmup")
            self._warmup_scheduler = GradualWarmupScheduler(
                self._optimizer, multiplier=1,
                total_epoch=self._cfg['warmup_epochs'],
                after_scheduler=self._scheduler
            )

    def _train_task(self, train_loader, val_loader):
        _, _, train_loader = self._inc_dataset.get_custom_loader(
            list(range(self._n_classes)), mode="train",
        )
        _, _, val_loader = self._inc_dataset.get_custom_loader(
            list(range(self._n_classes)), data_source="val",
        )
        logger.info(f"nb {len(train_loader.dataset)}")

        accu = ClassErrorMeter(accuracy=True, topk=[1])

        self._optimizer.zero_grad()
        self._optimizer.step()

        for epoch in range(self._n_epochs):
            _loss = 0.0
            accu.reset()
            if self._warmup:
                self._warmup_scheduler.step()
                if epoch == self._cfg['warmup_epochs']:
                    self._network.classifier.reset_parameters()

            for i, (inputs, targets) in enumerate(train_loader, start=1):
                self.train()
                self._optimizer.zero_grad()

                loss_ce = self._forward_loss(inputs, targets, accu)
                loss = loss_ce

                if not utils.check_loss(loss):
                    import pdb
                    pdb.set_trace()

                loss.backward()
                self._optimizer.step()

                _loss += loss_ce.item()

            if not self._warmup:
                self._scheduler.step()

            logger.info(
                "Task {}/{}, Epoch {}/{} => Clf loss: {}, Train Accu: {}".
                format(
                    self._task + 1,
                    self._n_tasks,
                    epoch + 1,
                    self._n_epochs,
                    round(_loss / i, 3),
                    round(accu.value()[0], 3),
                )
            )
            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(val_loader)

    def _forward_loss(self, inputs, targets, accu=None):
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)

        outputs = self._parallel_network(inputs)  # -> logits: [B, T, N] features: [B, F]
        assert len(outputs['logit'].shape) == 3 and outputs['logit'].shape[1] == self._cfg['T']  # -> check if the T is correct
        freq_opts = torch.mean(outputs['logit'], dim=1)  # -> [B, N]
        if accu is not None:
            accu.add(freq_opts, targets)

        if self.TET_config['enable']:
            loss = utils.TET_loss(outputs['logit'], targets, self.TET_config['TET_mean'], self.TET_config['TET_lamb'])
        else:
            loss = F.cross_entropy(freq_opts, targets)

        ce_loss = loss

        return ce_loss

    def _after_task(self, taski, inc_dataset):
        network = deepcopy(self._parallel_network)
        network.eval()

        logger.info("save model")
        cur_save_path = f"{self.save_path}/step{self._task}.ckpt"

        if self._cfg["save_ckpt"] and taski >= self._cfg["start_task"]:
            torch.save(network.cpu().state_dict(), cur_save_path)

        # -> a finetune part maybe...
        # -> a possible analysing part
        if self._analyse:
            self.analyse_plotting()
        ###############################
        # -> a part of old model saving...

    def _eval_task(self, data_loader):
        self._parallel_network.eval()
        ypred, ytrue = [], []

        for idx, (inputs, targets) in enumerate(data_loader):
            with torch.no_grad():
                #######################################################
                logits = self._network(inputs.to(self._device))["logit"]  # -> [B, T, N]
                logits = torch.mean(logits, dim=1)  # -> [B, N]
                #######################################################

            ytrue.append(targets.numpy())
            ypred.append(torch.softmax(logits, dim=1).cpu().numpy())

        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)

        return ypred, ytrue

    def load_cur_model(self):
        load_path = f"{self.save_path}/step{self._task}.ckpt"
        assert os.path.exists(load_path), "The corresponding task step has not existed"
        self._parallel_network.load_state_dict(torch.load(load_path))

    def validate(self, data_loader):
        ypred, ytrue = self._eval_task(data_loader)
        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=self._increments, n_classes=self._n_classes)
        return test_acc_stats['top1']['total']

    def analyse_plotting(self):
        from utils.utils import get_feature_logit, get_spiking_temporal_logits
        _, _, down_loader = self._inc_dataset.get_custom_loader(
            list(range(self._n_classes)), data_source="train", mode="downsample_train"
        )
        logger.info(f"nb {len(down_loader.dataset)}")
        get_feature_logit(
            logger=logger, eb_network=self._network, cur_n_cls=self._n_classes,
            loader=down_loader, device=self._device, save_dir=self.save_path,
            taski=self._task, loader_label="single"
        )
        get_spiking_temporal_logits(
            logger=logger, net=self._network, cur_n_cls=self._n_classes,
            loader=down_loader, device=self._device, save_dir=self.save_path,
            taski=self._task
        )



