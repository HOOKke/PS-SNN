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
from utils.utils import update_classes_mean, finetune_spiking_classifier

# Constants
EPSILON = 1e-8
logger = logging.getLogger(__name__)


# -> final type of the model...
class Spiking_Mutable2(IncrementalLearner):
    effect_fac = 8
    change_fac = 4

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
        self._network = network.Spiking_BasicNet_eb(
            cfg['convnet'], cfg=cfg,
            device=self._device
        )

        self._parallel_network = DataParallel(self._network, self._multi_devices)
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
        self.TET_config = cfg['TET_config']
        logger.info(f"TET_config: {self.TET_config}")

        self._temperature = cfg["temperature"]
        self._distillation = cfg["distillation"]
        # other setting
        self.use_aux = cfg['use_aux']  # -> whether to use aux_loss
        self.supp_o2n = cfg['supp_o2n']  # -> whether to suppress the o2n weight part
        self.step_kl = cfg['step_kl']  # -> whether consider all time-step when use distillation...
        self.kl_config = cfg['kl_config']  # -> whether distillation old
        assert self.kl_config in ['none', 'all_sample', 'old_sample']
        logger.info(f"step_kl: {self.step_kl}")
        logger.info(f"use_aux: {self.use_aux}, aux_nplus1: {self._cfg['aux_n+1']}")
        logger.info(f"supp_o2n: {self.supp_o2n}, use_full_kl: {self.kl_config}")
        # Classifier Learning Stage
        self._decouple = cfg["decouple"]

    def eval(self):
        self._parallel_network.eval()

    def train(self):
        self._parallel_network.train()
        self._parallel_network.module.convnets[-1].train()
        if self._task >= 1:
            for i in range(self._task):
                self._parallel_network.module.convnets[i].eval()

    def _before_task(self, taski, val_loader):
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
        logger.info(f"Now {self._memory_per_class} exemplars per class.")

        self._network.add_classes(self._task_size)
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

        self._optimizer = factory.get_optimizer(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            self._opt_name, lr, weight_decay
        )

        if "cos" in self._cfg["scheduler"]:
            logger.info("The scheduler is CosineAnnealing")
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self._n_epochs, eta_min=0)
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
        if self._task == 0:
            self._train_first_task(train_loader, val_loader)
        else:
            self._train_latter_task(train_loader, val_loader)

        # # ImageNet part ########################################
        # # For the large-scale dataset, we manage the data in the shared memory. (which may be used in the herding part)
        # self._inc_dataset.shared_data_inc = train_loader.dataset.share_memory
        # ########################################################

    def _train_first_task(self, train_loader, val_loader):
        assert self._task == 0
        topk = 5 if self._n_classes > 5 else self._task_size
        accu = ClassErrorMeter(accuracy=True, topk=[1, topk])

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

                loss_ce = self._compute_first_loss(inputs, targets, accu)
                loss = loss_ce

                loss.backward()
                self._optimizer.step()

                _loss += loss_ce.item()

            if not self._warmup:
                self._scheduler.step()

            if (epoch + 1) % 3 == 0:
                logger.info(
                    "Task {}/{}, Epoch {}/{} => Clf loss: {}, Train Accu: {}".
                    format(
                        self._task + 1,
                        self._n_tasks,
                        epoch + 1,
                        self._n_epochs,
                        round(_loss / i, 2),
                        round(accu.value()[0], 2),
                    )
                )

            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(val_loader)

    def _compute_first_loss(self, inputs, targets, accu=None):
        assert self._task == 0
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)

        outputs = self._parallel_network(inputs)
        assert outputs['logit'].shape[1] == self._cfg['T']
        freq_opts = torch.mean(outputs['logit'], dim=1)

        if accu is not None:
            accu.add(freq_opts, targets)

        if self.TET_config['enable']:
            loss = utils.TET_loss(outputs['logit'], targets, self.TET_config['TET_mean'], self.TET_config['TET_lamb'])
        else:
            loss = F.cross_entropy(freq_opts, targets)

        return loss

    def _train_latter_task(self, train_loader, val_loader):
        topk = 5 if self._n_classes > 5 else self._task_size
        accu = ClassErrorMeter(accuracy=True, topk=[1, topk])

        self._optimizer.zero_grad()
        self._optimizer.step()

        ######################################
        for epoch in range(self._n_epochs):
            _loss_ce, _loss_aux, _loss_extra = 0.0, 0.0, 0.0
            accu.reset()
            if self._warmup:
                self._warmup_scheduler.step()
                if epoch == self._cfg['warmup_epochs']:
                    self._network.classifier.reset_parameters()
                    self._network.aux_clf.reset_parameters()

            for i, (inputs, targets) in enumerate(train_loader):
                self.train()
                self._optimizer.zero_grad()
                old_classes = targets < (self._n_classes - self._task_size)
                new_classes = targets >= (self._n_classes - self._task_size)
                loss_ce, loss_aux, loss_ex = self._compute_combined_loss(
                    inputs, targets, old_classes, new_classes,
                    epoch=epoch, accu=accu,
                )
                loss = loss_ce + loss_aux + loss_ex

                if not utils.check_loss(loss):
                    import pdb
                    pdb.set_trace()

                loss.backward()
                self._optimizer.step()

                _loss_ce += loss_ce.item()
                _loss_aux += loss_aux.item()
                _loss_extra += loss_ex.item()

                # Some operation after each time the weight change #####################
                # if self.fix_o2o:
                #     self._network.reset_old_clf_weight(self._old_model.module.clf_weight)
                if self.supp_o2n:
                    self._network.parameter_constraint("o2n")
                ########################################################################

            if not self._warmup:
                self._scheduler.step()

            if (epoch + 1) % 3 == 0:
                logger.info(
                    "Task {}/{}, Epoch {}/{} => Clf loss: {}, Aux loss: {}, Extra loss: {}, Train Accu: {}".
                    format(
                        self._task + 1,
                        self._n_tasks,
                        epoch + 1,
                        self._n_epochs,
                        round(_loss_ce / i, 3),
                        round(_loss_aux / i, 3),
                        round(_loss_extra / i, 3),
                        round(accu.value()[0], 3),
                    ))
            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(val_loader)

    def _compute_combined_loss(
        self, inputs, targets, old_classes, new_classes,
        epoch, accu=None
        # -> maybe will have others... though
    ):
        assert self._task > 0
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)

        outputs = self._parallel_network(inputs)
        freq_opts = torch.mean(outputs['logit'], dim=1)  # [B, T, N] -> [B, N]
        if accu is not None:
            accu.add(freq_opts, targets)

        ###########################################################
        # The part of the ce loss distribution
        #######################################################
        ce_loss = self._auged_ce_loss(
            inputs, outputs, 'logit', targets, epoch, old_classes, new_classes, kl_config=self.kl_config
        )

        #######################################################
        # The part of the aux loss
        #######################################################
        if self.use_aux:
            # POSSIBLE AUGMENT THOUGH ######
            aux_target = targets.clone()
            if self._cfg['aux_n+1']:
                aux_target[old_classes] = 0
                aux_target[new_classes] -= sum(self._inc_dataset.increments[:self._task]) - 1

            aux_loss = self._auged_ce_loss(
                inputs, outputs, 'aux_logit', aux_target, epoch, old_classes, new_classes, kl_config="none"
            )
        else:
            aux_loss = torch.zeros([1]).to(self._device)

        #######################################################
        # The part of the extra loss
        #######################################################
        extra_loss = torch.zeros([1])[0].to(self._device)

        return ce_loss, aux_loss, extra_loss

    def _after_task(self, taski, inc_dataset):
        network = deepcopy(self._parallel_network)
        network.eval()

        logger.info("save model")
        cur_save_path = f"{self.save_path}/step{self._task}.ckpt"
        if self._cfg["save_ckpt"] and taski >= self._cfg["start_task"]:
            torch.save(network.cpu().state_dict(), cur_save_path)

        if self._cfg['decouple']['enable'] and taski > 0:
            finetune_spiking_classifier(
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
                new_cls_size=self._task_size,
                refresh_param=self._decouple['refresh_param'],
                loader_type=self._decouple.get('loader_type', 'balanced'),
                logit_ctrl=self._decouple.get('logit_ctrl', False),
            )
        #################################################
        # Here consider a feature distribution plotting (no use for now)
        #################################################
        # self.analyse_plotting()
        if self._cfg["infer_head"] == 'NCM':
            logger.info("compute prototype")
            self.update_prototype()

        # Memory update
        self._memory_size.update_n_classes(self._n_classes)
        self._memory_size.update_memory_per_cls(self._network, self._n_classes, self._task_size)
        logger.info(f"Now {self._memory_per_class} examplars per class.")

        if self._memory_size.memsize != 0:
            logger.info("build memory")
            self.build_exemplars(inc_dataset, self._coreset_strategy)

            # The code to save the memory is useless in experiments...
            # if self._cfg["save_mem"]:
            #     mem_dir = os.path.join(self.save_path, "mem")
            #     memory = {
            #         'x': inc_dataset.data_memory,
            #         'y': inc_dataset.targets_memory,
            #         'herding': self._herding_matrix
            #     }
            #     if not os.path.exists(mem_dir):
            #         os.makedirs(mem_dir)
            #     mem_ckpt_path = os.path.join(mem_dir, f"mem_step{self._task}.ckpt")
            #     if not (os.path.exists(mem_ckpt_path)) and self._cfg["save_mem"]:
            #         torch.save(memory, mem_ckpt_path)
            #         logger.info(f"Save step{self._task} memory!")

        self._parallel_network.eval()
        self._old_model = deepcopy(self._parallel_network)
        self._old_model.module.freeze()

        # # ImageNet part ########################################
        # del self._inc_dataset.shared_data_inc
        # self._inc_dataset.shared_data_inc = None
        # ########################################################

    def _eval_task(self, data_loader):
        ypred, ytrue = self._compute_accuracy_by_netout(data_loader)

        return ypred, ytrue

    def _compute_accuracy_by_netout(self, data_loader):
        self._parallel_network.eval()
        preds, targets = [], []

        with torch.no_grad():
            for i, (inputs, lbls) in enumerate(data_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                #######################################################
                _logits = self._parallel_network(inputs)['logit']  # -> [B, T, N]
                _preds = torch.mean(_logits, dim=1)
                #######################################################
                preds.append(_preds.detach().cpu().numpy())
                targets.append(lbls.long().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        return preds, targets

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
            # # ImageNet part ########################################
            # data_inc = self._inc_dataset.shared_data_inc if self._inc_dataset.shared_data_inc is not None else self._inc_dataset.data_inc
            # ########################################################
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

    def validate(self, data_loader):
        ypred, ytrue = self._eval_task(data_loader)
        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=self._increments, n_classes=self._n_classes)
        logger.info(f"test top1acc:{test_acc_stats['top1']}")
        return test_acc_stats['top1']['total']

    def update_prototype(self):
        self._class_means = update_classes_mean(self._parallel_network,
                                                self._inc_dataset,
                                                self._n_classes,
                                                self._task_size,
                                                share_memory=self._inc_dataset.shared_data_inc,
                                                metric='None')

    def _auged_ce_loss(
        self, inputs, outputs, opt_label, targets, epoch,
        old_classes, new_clasees, kl_config,
    ):
        assert len(inputs.shape) == 4 and len(outputs[opt_label].shape) == 3, f"output's shape is {outputs[opt_label].shape}"
        if kl_config == 'all_sample':
            kl_mask = torch.ones_like(old_classes).to(self._device, torch.bool)
        elif kl_config == 'old_sample':
            kl_mask = old_classes
        elif kl_config == 'none':
            kl_mask = torch.zeros_like(old_classes).to(self._device, torch.bool)
        else:
            raise ValueError(f"<{self.kl_config}> is not a valid config for kl")

        if torch.sum(kl_mask) > 0:
            with torch.no_grad():
                old_logit = self._old_model(inputs[kl_mask])['logit'].detach()
            new_logit = outputs[opt_label].clone()[kl_mask][..., :-self._task_size]
            # -> consider whether i need to use the frequency representations
            if not self.step_kl:  # -> use frequency to distillation though...
                old_logit = torch.mean(old_logit, dim=1)  # [B, T, N] -> [B, N]
                new_logit = torch.mean(new_logit, dim=1)  # [B, T, N] -> [B, N]
                assert len(old_logit) == len(new_logit)
                assert old_logit.shape[-1] == new_logit.shape[-1] == self._n_classes - self._task_size
            #################################################################
            kl_loss = F.kl_div(
                (new_logit/self._temperature).log_softmax(dim=-1),
                (old_logit/self._temperature).softmax(dim=-1), reduction='none'
            )  # -> [B, T, N]
            if not self.step_kl:
                kl_loss = torch.sum(kl_loss, dim=-1)  # [B, N] -> [B]
            else:
                kl_loss = torch.mean(torch.sum(kl_loss, dim=-1), dim=1)  # [B, T, N] -> [B, T] -> [B]

        # -> ce_part #####
        # POSSIBLE AUGMENT THOUGH ######
        logit_opts = outputs[opt_label]  # -> [B, T, N]

        if self.TET_config['enable']:
            ce_loss = utils.TET_loss(logit_opts, targets, self.TET_config['TET_mean'], self.TET_config['TET_lamb'], reduction='none')
        else:
            mean_opts = torch.mean(logit_opts, dim=1)  # [B, T, N] -> [B, N]
            ce_loss = F.cross_entropy(mean_opts, targets, reduction='none')

        if torch.sum(kl_mask) > 0:
            ce_loss[kl_mask] += kl_loss
        ce_loss = torch.mean(ce_loss)
        return ce_loss

    @property
    def _memory_per_class(self):
        return self._memory_size.mem_per_cls

    def analyse_plotting(self):
        if self._task == 1 or self._task == 2:
            logger.info(f"Generate the steps logit at task-{self._task}")
            # _, _, test_loader = self._inc_dataset.get_custom_loader(
            #     list(range(self._n_classes)), mode="test", data_source="test"
            # )
            # _, _, train_loader = self._inc_dataset.get_custom_loader(
            #     list(range(self._n_classes)), mode="test", data_source="train"
            # )
            # sample_num = 50
            part_loader = self._inc_dataset._get_loader(
                self._inc_dataset.data_inc, self._inc_dataset.targets_inc, mode="balanced_train"
            )
            sample_num = 30
            utils.get_step_logits(
                logger=logger, net=self._network, cur_n_cls=self._n_classes,
                loader=part_loader, device=self._device,
                save_dir=self.save_path, taski=self._task
            )
