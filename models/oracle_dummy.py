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
from utils.network.memory import MemorySetting
from utils.schedulers import GradualWarmupScheduler

# Constants
EPSILON = 1e-8
logger = logging.getLogger(__name__)


class Oracle_Dummy(IncrementalLearner):
    def __init__(self, cfg, inc_dataset=None, tensorboard=None, ):
        super().__init__()
        self._cfg = cfg
        self._device = cfg['device'][0]
        self._multi_devices = cfg['device']

        ###########################################
        # Network settings
        self._network = network.BasicNet_dummy(
            cfg["convnet"],
            cfg=cfg,
            device=self._device,
            use_bias=cfg["use_bias"],
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
        self._decouple = cfg.get("decouple", None)

        self._analyse = cfg.get('analyse', False)
        if self._analyse:
            logger.info("conduct several analysing in this epoch of experiment")

        ###########################################
        # Examplar settings
        self._memory_size = MemorySetting(
            mode=cfg["mem_size_mode"], total_memory=cfg["memory_size"],
            fixed_memory_per_cls=cfg["fixed_memory_per_cls"]
        )
        self._herding_matrix = []
        self._coreset_strategy = cfg["coreset_strategy"]


    def eval(self):
        self._parallel_network.eval()

    def train(self):
        self._parallel_network.train()

    def _before_task(self, taski, inc_dataset):
        self._task = taski
        self._n_classes += self._task_size

        if taski == 0 and self._init_config != {}:
            logger.info(f"Take special setting for the first trail: {self._init_config}")
            self._n_epochs = self._init_config.get('epochs', self._cfg["epochs"])
            self._lr = self._init_config.get('lr', self._cfg["lr"])
            self._lr_decay = self._init_config.get('lr_decay', self._cfg["lr_decay"])
            self._opt_name = self._init_config.get('optimizer', self._cfg["optimizer"])
            self._scheduling = self._init_config.get('scheduling', self._cfg["scheduling"])
            self._weight_decay = self._init_config.get('weight_decay', self._cfg["weight_decay"])
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

            if (epoch + 1) % 5 == 0:
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

        outputs = self._parallel_network(inputs)

        if accu is not None:
            accu.add(outputs['logit'], targets)

        ce_loss = F.cross_entropy(outputs['logit'], targets)

        return ce_loss

    def _after_task(self, taski, inc_dataset):
        network = deepcopy(self._parallel_network)
        network.eval()

        logger.info("save model")
        cur_save_path = f"{self.save_path}/step{self._task}.ckpt"

        if self._cfg["save_ckpt"] and taski >= self._cfg["start_task"]:
            torch.save(network.cpu().state_dict(), cur_save_path)

        # -> a finetune part maybe...
        if taski > 0 and self._decouple is not None and self._decouple["enable"]:
            from utils.utils import custom_finetune_clf
            custom_finetune_clf(
                self._inc_dataset,
                logger,
                self._parallel_network,
                self._n_classes,
                nepoch=self._decouple["epochs"],
                lr=self._decouple["lr"],
                scheduling=self._decouple["scheduling"],
                lr_decay=self._decouple["lr_decay"],
                weight_decay=self._decouple["weight_decay"],
                temperature=self._decouple["temperature"],
                device=self._device,
                refresh_param=self._decouple['refresh_param'],
                loader_type=self._decouple.get('loader_type', 'balanced'),
            )

        # -> a possible analysing part
        if self._analyse:
            self.analyse_plotting()

        # -> memory updating part
        self._memory_size.update_n_classes(self._n_classes)
        self._memory_size.update_memory_per_cls(self._network, self._n_classes, self._task_size)
        logger.info(f"Now {self._memory_per_class} examplars per class.")

        if self._memory_size.memsize != 0:
            logger.info("build memory")
            self.build_exemplars(inc_dataset, self._coreset_strategy)

    def _eval_task(self, data_loader):
        self._parallel_network.eval()
        ypred, ytrue = [], []

        for idx, (inputs, targets) in enumerate(data_loader):
            with torch.no_grad():
                logits = self._network(inputs.to(self._device))["logit"]

            ytrue.append(targets.numpy())
            ypred.append(torch.softmax(logits, dim=1).cpu().numpy())

        ytrue = np.concatenate(ytrue, axis=0)
        ypred = np.concatenate(ypred, axis=0)

        # if self._analyse:
        #     if self._n_classes > self._task_size:
        #         logger.info(f"Start check the new-old binary distinguish ability")
        #         from utils.utils import binary_finetune, binary_weight_check
        #         binary_finetune(
        #             inc_dataset=self._inc_dataset, logger=logger, network=self._parallel_network,
        #             n_class=self._n_classes, task_size=self._task_size, taski=self._task, save_dir=self.save_path,
        #             nepoch=20, lr=0.1, scheduling=[10], lr_decay=0.1, weight_decay=5e-4, device=self._device
        #         )
        #         binary_weight_check(self._network.classifier.weight.data[0, :], out_dim=self._network.out_dim)

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
        from utils.utils import get_feature_logit
        _, _, down_loader = self._inc_dataset.get_custom_loader(
            list(range(self._n_classes)), data_source="train", mode="downsample_train",
        )
        logger.info(f"nb {len(down_loader.dataset)}")
        get_feature_logit(
            logger=logger, eb_network=self._network, cur_n_cls=self._n_classes,
            loader=down_loader, device=self._device, save_dir=self.save_path,
            taski=self._task, loader_label="single"
        )

        balanced_loader = self._inc_dataset._get_loader(
            self._inc_dataset.data_inc, self._inc_dataset.targets_inc,
            mode="balanced_train"
        )
        logger.info(f"nb {len(balanced_loader.dataset)}")
        get_feature_logit(
            logger=logger, eb_network=self._network, cur_n_cls=self._n_classes,
            loader=balanced_loader, device=self._device, save_dir=self.save_path,
            taski=self._task, loader_label="inc_data"
        )

        if self._task > 0:
            old_weight_norm = 0
            new_weight_norm = 0

            for i in range(self._n_classes - self._task_size):
                old_weight_norm += torch.norm(self._network.classifier.weight[i, :])
            for i in range(self._n_classes - self._task_size, self._n_classes):
                new_weight_norm += torch.norm(self._network.classifier.weight[i, :])
            old_weight_norm /= (self._n_classes - self._task_size)
            new_weight_norm /= self._task_size
            logger.info(f"old_weight_norm is {old_weight_norm:.3f}, new_weight_norm is {new_weight_norm:.3f}")

        # weight_norm = torch.norm(self._network.classifier.weight)
        # for i in range(self._n_classes):
        #     weight_norm.append(torch.mean(torch.abs(self._network.classifier.weight[i, :])).item())
        # logger.info(f"{weight_norm}")
        # logger.info(f"mean_weight_norm: {torch.mean(torch.Tensor(weight_norm)).item()}")

    def build_exemplars(self, inc_dataset, coreset_strategy):
        save_path = os.path.join(self.save_path, f"mem/mem_step{self._task}.ckpt")
        if self._cfg["load_mem"] and os.path.exists(save_path):
            memory_states = torch.load(save_path)
            self._inc_dataset.data_memory = memory_states['x']
            self._inc_dataset.targets_memory = memory_states['y']
            self._herding_matrix = memory_states['herding']
            logger.info(f"Load saved step{self._task} memory!")
            return

        if coreset_strategy == "random":
            from utils.network.memory import random_selection
            self._inc_dataset.data_memory, self._inc_dataset.targets_memory = random_selection(
                n_classes=self._n_classes,
                task_size=self._task_size,
                network=self._parallel_network,
                logger=logger,
                inc_dataset=inc_dataset,
                memory_per_class=self._memory_per_class,
            )
        elif coreset_strategy == "iCaRL":
            from utils.network.memory import herding
            # data_inc = self._inc_dataset.shared_data_inc if self._inc_dataset.shared_data_inc is not None else self._inc_dataset.data_inc
            data_inc = inc_dataset.data_inc
            self._inc_dataset.data_memory, self._inc_dataset.targets_memory, self._herding_matrix = herding(
                n_classes=self._n_classes,
                task_size=self._task_size,
                network=self._parallel_network,
                herding_matrix=self._herding_matrix,
                inc_dataset=inc_dataset,
                shared_data_inc=data_inc,
                memory_per_class=self._memory_per_class,
                logger=logger,
            )
        else:
            raise ValueError()

    @property
    def _memory_per_class(self):
        return self._memory_size.mem_per_cls



