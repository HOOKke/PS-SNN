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

import utils.utils
from utils import network, factory, utils
from models import IncrementalLearner
from utils.metrics import ClassErrorMeter
from utils.network.memory import MemorySetting
from utils.schedulers import GradualWarmupScheduler
from utils.utils import extract_features, update_classes_mean
from utils.utils import custom_finetune_clf

# Constants
EPSILON = 1e-8
logger = logging.getLogger(__name__)


class IncModel(IncrementalLearner):
    def __init__(self, cfg, inc_dataset=None, tensorboard=None):
        super().__init__()
        self._cfg = cfg
        self._device = cfg['device'][0]
        self._multi_devices = cfg['device']

        ###########################################
        # Examplar settings
        self._memory_size = MemorySetting(
            mode=cfg["mem_size_mode"], total_memory=cfg["memory_size"],
            fixed_memory_per_cls=cfg["fixed_memory_per_cls"]
        )
        self._herding_matrix = []
        self._coreset_strategy = cfg["coreset_strategy"]

        ###########################################
        # Network settings
        self._der = cfg['der']  # Whether to expand the representation
        self._network = network.BasicNet_der(
            cfg["convnet"],
            cfg=cfg,
            device=self._device,
            use_bias=cfg["use_bias"],
        )
        self._parallel_network = DataParallel(self._network, self._multi_devices)
        self._infer_head = cfg["infer_head"]
        self._old_model = None

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
            if cfg["save_mem"]:
                save_path = os.path.join(ckpt_dir, "mem")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

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
        self._temperature = cfg["temperature"]
        self._distillation = cfg["distillation"]
        # Classifier Learning Stage
        self._decouple = cfg["decouple"]
        # fixation on my own
        self._loss_type = cfg.get('loss_type', 'ce')  # ->can be [none, reg, ce]
        logger.info(f"Loss type is {self._loss_type}")

        self._analyse = cfg.get('analyse', False)
        if self._analyse:
            logger.info("conduct several analysing in this epoch of experiment")

    def eval(self):
        self._parallel_network.eval()

    def train(self):
        if self._der:
            self._parallel_network.train()
            self._parallel_network.module.convnets[-1].train()
            if self._task >= 1:
                for i in range(self._task):
                    self._parallel_network.module.convnets[i].eval()
        else:
            self._parallel_network.train()

    def _before_task(self, taski, inc_dataset):
        logger.info(f"Begin step {taski}")

        # Update Task info
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

        # Memory
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        self._network.add_classes(self._task_size)
        # self._network.task_size = self._task_size

        # from thop import profile
        # from thop import clever_format
        # dummy_input = torch.randn(1, 3, 32, 32).to(self._device)
        # flops, params = profile(self._network.convnets[0], inputs=(dummy_input, ))
        # flops, params = clever_format([flops, params], "%.3f")
        # print(f"parameters: {params}")
        # print(f"Flops of single sample: {flops}")

        # total = sum([param.nelement() for param in self._network.parameters()])
        # logger.info(f"The other calculation get a result parameters: {total}")
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

        if self._der and self._task > 0:
            for i in range(self._task):
                for p in self._parallel_network.module.convnets[i].parameters():
                    p.requires_grad = False

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
        logger.info(f"nb {len(train_loader.dataset)}")

        topk = 5 if self._n_classes > 5 else self._task_size
        accu = ClassErrorMeter(accuracy=True, topk=[1, topk])
        train_new_accu = ClassErrorMeter(accuracy=True)
        train_old_accu = ClassErrorMeter(accuracy=True)

        utils.display_weight_norm(logger, self._parallel_network, self._increments, "Initial trainset")
        utils.display_feature_norm(logger, self._parallel_network, train_loader, self._n_classes,
                                   self._increments, "Initial trainset", device=self._device)

        self._optimizer.zero_grad()
        self._optimizer.step()

        for epoch in range(self._n_epochs):
            _loss, _loss_aux = 0.0, 0.0
            accu.reset()
            train_new_accu.reset()
            train_old_accu.reset()
            if self._warmup:
                self._warmup_scheduler.step()
                if epoch == self._cfg['warmup_epochs']:
                    self._network.classifier.reset_parameters()
                    if self._cfg['use_aux_cls']:
                        self._network.aux_classifier.reset_parameters()
            for i, (inputs, targets) in enumerate(train_loader, start=1):
                self.train()
                self._optimizer.zero_grad()
                old_classes = targets < (self._n_classes - self._task_size)
                new_classes = targets >= (self._n_classes - self._task_size)
                loss_ce, loss_aux = self._forward_loss(
                    inputs,
                    targets,
                    old_classes,
                    new_classes,
                    accu=accu,
                    new_accu=train_new_accu,
                    old_accu=train_old_accu,
                )

                if self._cfg["use_aux_cls"] and self._task > 0:
                    loss = loss_ce + loss_aux
                else:
                    loss = loss_ce

                if not utils.check_loss(loss):
                    import pdb
                    pdb.set_trace()

                loss.backward()
                self._optimizer.step()

                if self._cfg["postprocessor"]["enable"]:
                    if self._cfg["postprocessor"]["type"].lower() == "wa":
                        for p in self._network.classifier.parameters():
                            p.data.clamp_(0.0)

                _loss += loss_ce.item()
                _loss_aux += loss_aux.item()

            if not self._warmup:
                self._scheduler.step()

            if (epoch + 1) % 3 == 0:
                logger.info(
                    "Task {}/{}, Epoch {}/{} => Clf loss: {} Aux loss: {}, Train Accu: {}, Train@5 Acc: {}, old acc:{}".
                    format(
                        self._task + 1,
                        self._n_tasks,
                        epoch + 1,
                        self._n_epochs,
                        round(_loss / i, 3),
                        round(_loss_aux / i, 3),
                        round(accu.value()[0], 3),
                        round(accu.value()[1], 3),
                        round(train_old_accu.value()[0], 3),
                    ))

            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(val_loader)

        # # For the large-scale dataset, we manage the data in the shared memory.
        # self._inc_dataset.shared_data_inc = train_loader.dataset.share_memory

        utils.display_weight_norm(logger, self._parallel_network, self._increments, "After training")
        utils.display_feature_norm(logger, self._parallel_network, train_loader, self._n_classes,
                                   self._increments, "Trainset", device=self._device)

    def _forward_loss(self, inputs, targets, old_classes, new_classes, accu=None, new_accu=None, old_accu=None):
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)

        outputs = self._parallel_network(inputs)
        if accu is not None:
            if self._task == 0 or self._loss_type != 'none':
                accu.add(outputs['logit'], targets)

        return self._compute_loss(inputs, targets, outputs, old_classes, new_classes)

    def _compute_loss(self, inputs, targets, outputs, old_classes, new_classes):
        if self._loss_type != "none":
            if self._loss_type == 'ce':
                loss = F.cross_entropy(outputs['logit'], targets)
            else:
                raise NotImplementedError(f"{self._loss_type} is not a valid loss type")
        else:
            loss = torch.zeros([1]).to(self._device)

        if outputs['aux_logit'] is not None:
            aux_targets = targets.clone()
            if self._cfg["aux_n+1"]:
                aux_targets[old_classes] = 0
                aux_targets[new_classes] -= sum(self._inc_dataset.increments[:self._task]) - 1
            aux_loss = F.cross_entropy(outputs['aux_logit'], aux_targets)
            ####################
            # aux_loss = F.cross_entropy(outputs['aux_logit'], aux_targets)
        else:
            aux_loss = torch.zeros([1]).to(self._device)

        return loss, aux_loss

    def _after_task(self, taski, inc_dataset):
        network = deepcopy(self._parallel_network)
        network.eval()

        logger.info("save model")
        cur_save_path = f"{self.save_path}/step{self._task}.ckpt"
        if self._cfg["save_ckpt"] and taski >= self._cfg["start_task"]:
            torch.save(network.cpu().state_dict(), cur_save_path)

        if self._cfg["decouple"]['enable'] and taski > 0:
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

            utils.display_weight_norm(logger, self._parallel_network, self._increments, "finetuning trainset")
            utils.check_clf_weight_distrib(self._parallel_network, self._task_size, self._network.out_dim, logger)

            # old_weight_norm = 0
            # new_weight_norm = 0
            # for i in range(self._n_classes - self._task_size):
            #     old_weight_norm += torch.norm(self._network.classifier.weight[i, :])
            # for i in range(self._n_classes - self._task_size, self._n_classes):
            #     new_weight_norm += torch.norm(self._network.classifier.weight[i, :])
            # old_weight_norm /= (self._n_classes - self._task_size)
            # new_weight_norm /= self._task_size
            # self.weight_align = old_weight_norm / new_weight_norm
            # logger.info(f"weight_align is {self.weight_align}")

        if self._cfg["postprocessor"]["enable"]:
            self._update_postprocessor(inc_dataset)

        if self._cfg["infer_head"] == 'NCM':
            logger.info("compute prototype")
            self.update_prototype()

        ###############################################
        # THE GROUP OF CODE WHICH USED TO ANALYSE THE FEATURES
        ###############################################
        if self._analyse:
            self.analyse_plotting()

        # Memory update
        self._memory_size.update_n_classes(self._n_classes)
        self._memory_size.update_memory_per_cls(self._network, self._n_classes, self._task_size)
        logger.info(f"Now {self._memory_per_class} examplars per class.")

        if self._memory_size.memsize != 0:
            logger.info("build memory")
            self.build_exemplars(inc_dataset, self._coreset_strategy)

            if self._cfg["save_mem"]:
                mem_dir = os.path.join(self.save_path, "mem")
                memory = {
                    'x': inc_dataset.data_memory,
                    'y': inc_dataset.targets_memory,
                    'herding': self._herding_matrix
                }
                if not os.path.exists(mem_dir):
                    os.makedirs(mem_dir)
                mem_ckpt_path = os.path.join(mem_dir, f"mem_step{self._task}.ckpt")
                if not (os.path.exists(mem_ckpt_path)) and self._cfg["save_mem"]:
                    torch.save(memory, mem_ckpt_path)
                    logger.info(f"Save step{self._task} memory!")

        self._parallel_network.eval()
        self._old_model = deepcopy(self._parallel_network)
        self._old_model.module.freeze()
        # del self._inc_dataset.shared_data_inc
        # self._inc_dataset.shared_data_inc = None

    def _eval_task(self, data_loader):
        if self._infer_head == "softmax":
            ypred, ytrue = self._compute_accuracy_by_netout(data_loader)
        elif self._infer_head == "NCM":
            ypred, ytrue = self._compute_accuracy_by_ncm(data_loader)
        else:
            raise ValueError()

        if self._analyse:
            if self._n_classes > self._task_size:
                logger.info(f"Start check the new-old binary distinguish ability")
                from utils.utils import binary_finetune
                binary_finetune(
                    inc_dataset=self._inc_dataset, logger=logger, network=self._parallel_network,
                    n_class=self._n_classes, task_size=self._task_size, taski=self._task, save_dir=self.save_path,
                    nepoch=20, lr=0.1, scheduling=[10], lr_decay=0.1, weight_decay=5e-4, device=self._device
                )

        return ypred, ytrue

    def _compute_accuracy_by_netout(self, data_loader):
        preds, targets = [], []
        self._parallel_network.eval()
        with torch.no_grad():
            for i, (inputs, lbls) in enumerate(data_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                _preds = self._parallel_network(inputs)['logit']
                if self._cfg["postprocessor"]["enable"] and self._task > 0:
                    _preds = self._network.postprocessor.post_process(_preds, self._task_size)
                # if self._task > 0:
                #     _preds[:, -self._task_size:] *= self.weight_align
                preds.append(_preds.detach().cpu().numpy())
                targets.append(lbls.long().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        return preds, targets

    def _compute_accuracy_by_ncm(self, loader):
        features, targets_ = extract_features(self._parallel_network, loader)
        targets = np.zeros((targets_.shape[0], self._n_classes), np.float32)
        targets[range(len(targets_)), targets_.astype("int32")] = 1.0

        class_means = (self._class_means.T / (np.linalg.norm(self._class_means.T, axis=0) + EPSILON)).T

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T
        # Compute score for iCaRL
        sqd = cdist(class_means, features, "sqeuclidean")
        score_icarl = (-sqd).T
        return score_icarl[:, :self._n_classes], targets_

    def _update_postprocessor(self, inc_dataset):
        if self._cfg["postprocessor"]["type"].lower() == "bic":
            if self._cfg["postprocessor"]["disalign_resample"] is True:
                bic_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                     inc_dataset.targets_inc,
                                                     mode="train",
                                                     resample='disalign_resample')
            else:
                xdata, ydata = inc_dataset._select(inc_dataset.data_train,
                                                   inc_dataset.targets_train,
                                                   low_range=0,
                                                   high_range=self._n_classes)
                bic_loader = inc_dataset._get_loader(xdata, ydata, shuffle=True, mode='train')
            bic_loss = None
            self._network.postprocessor.reset(n_classes=self._n_classes)
            self._network.postprocessor.update(logger,
                                               self._task_size,
                                               self._parallel_network,
                                               bic_loader,
                                               loss_criterion=bic_loss)
        elif self._cfg["postprocessor"]["type"].lower() == "wa":
            logger.info("Post processor wa update !")
            self._network.postprocessor.update(self._network.classifier, self._task_size)

    def update_prototype(self):
        self._class_means = update_classes_mean(self._parallel_network,
                                                self._inc_dataset,
                                                self._n_classes,
                                                self._task_size,
                                                share_memory=self._inc_dataset.shared_data_inc,
                                                metric='None')

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

    def load_cur_model(self):
        load_path = f"{self.save_path}/step{self._task}.ckpt"
        assert os.path.exists(load_path), "The corresponding task step has not existed"
        self._parallel_network.load_state_dict(torch.load(load_path))

    def load_after_task_info(self, taski, inc_dataset):  # -> recover from saved information
        self._after_task(taski, inc_dataset)

    def validate(self, data_loader):
        if self._infer_head == 'NCM':
            self.update_prototype()
        ypred, ytrue = self._eval_task(data_loader)
        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=self._increments, n_classes=self._n_classes)
        # logger.info(f"test top1acc:{test_acc_stats['top1']}")
        return test_acc_stats['top1']['total']

    @property
    def _memory_per_class(self):
        return self._memory_size.mem_per_cls

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
