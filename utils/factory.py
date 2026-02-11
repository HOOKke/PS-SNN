import warnings

import torch
from torch import optim

import models
import data_utils
from utils import schedulers
from extractor import (
    my_resnet, resnet,
    # -> spiking part
    spiking_resnet, spiking_cifarresnet, spiking_resnet_modified, spiking_rebuffi
)


def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer == "sgd_nesterov":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)

    raise NotImplementedError


def get_convnet(convnet_type, **kwargs):
    # if kwargs.get('T', None) is not None:
    #     print(f"The number of timesteps is {kwargs['T']}")
    if convnet_type == "resnet18":
        return resnet.resnet18(**kwargs)
    elif convnet_type == "resnet34":
        return resnet.resnet34(**kwargs)
    elif convnet_type == "resnet101":
        return resnet.resnet101(**kwargs)
    elif convnet_type == "rebuffi":
        return my_resnet.resnet_rebuffi(**kwargs)
    # -> appended for new method
    elif convnet_type == "spiking_rebuffi":
        return spiking_rebuffi.spiking_rebuffi(**kwargs)
    elif convnet_type == "spiking_cifar_resnet":
        return spiking_cifarresnet.spiking_resnet_cifar(**kwargs)
    elif convnet_type == "spiking_resnet18":
        return spiking_resnet.spiking_resnet18(**kwargs)
    elif convnet_type == "spiking_resnet18_modified":
        return spiking_resnet_modified.spiking_resnet18(**kwargs)
    else:
        raise NotImplementedError("Unknown convnet type {}.".format(convnet_type))


def get_model(args, inc_dataset, tenboard):
    dict_models = {
        "incmodel": models.IncModel,
        # "incmodel_test": models.IncModel_TEST,
        "oracle_inc": models.Oracle_INC,
        "oracle_dummy": models.Oracle_Dummy,

        "spiking_mutable": models.Spiking_Mutable,
        "spiking_mutable2": models.Spiking_Mutable2,
        "spiking_oracle_dummy": models.Spiking_Oracle_Dummy,

        # "binary_eb": models.BINARY_EB,
        # "bal_binary": models.BAL_BINARY,
        # "bal_inc": models.BAL_INC,
    }

    model = args["model"].lower()

    if model not in dict_models:
        raise NotImplementedError(
            f"Unknown model {args['model']}, must be among {list(dict_models.keys())}"
        )

    return dict_models[model](args, inc_dataset, tenboard)


# -> Get the incremental dataset
def get_data(args, class_order=None):
    return data_utils.IncrementalDataset(
        dataset_name=args["dataset"],
        random_order=args["random_classes"],
        shuffle=True,
        batch_size=args["batch_size"],
        workers=args["workers"],
        validation_split=args["validation"],
        # onehot=args["onehot"],
        increment=args["increment"],
        initial_increment=args["initial_increment"],
        sampler=get_sampler(args),
        sampler_config=args.get("sampler_config", {}),
        data_path=args["data_path"],
        class_order=class_order,
        seed=args["seed"],
        dataset_transforms=args.get("dataset_transforms", {}),
        all_test_classes=args.get("all_test_classes", False),
        metadata_path=args.get("metadata_path")
    )


# -> set a device list
def set_device(args):
    devices = []

    for device_type in args["device"]:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{device_type}")

        devices.append(device)

    args["device"] = devices


# -> get the corresponding sampler
def get_sampler(args):
    if args["sampler"] is None:
        return None

    sampler_type = args["sampler"].lower().strip()

    if sampler_type == "npair":
        return data_utils.NPairSampler
    elif sampler_type == "triplet":
        return data_utils.TripletSampler
    elif sampler_type == "tripletsemihard":
        return data_utils.TripletCKSampler
    elif sampler_type == "over_mem":
        return data_utils.MemoryOverSampler
    elif sampler_type == "keep_q":
        return data_utils.KeepQuantitySampler

    return ValueError(f"Unknown sampler {sampler_type}")


def get_lr_scheduler(
    scheduling_config, optimizer, nb_epochs, lr_decay=0.1, warmup_config=None, task=0,
):
    if scheduling_config is None:
        return None
    elif isinstance(scheduling_config, str):
        warnings.warn("Use a dict not a string for scheduling config!", DeprecationWarning)
        scheduling_config = {"type": scheduling_config}
    elif isinstance(scheduling_config, list):
        warnings.warn("Use a dict not a list for scheduling config!", DeprecationWarning)
        scheduling_config = {"type": "step", "epochs": scheduling_config}

    if scheduling_config["type"] == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            scheduling_config['epochs'],
            gamma=scheduling_config.get("gamma") or lr_decay
        )
    elif scheduling_config["type"] == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, scheduling_config["gamma"])
    elif scheduling_config["type"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=scheduling_config["gamma"]
        )
    elif scheduling_config["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nb_epochs)
    elif scheduling_config["type"] == "cosine_with_restart":
        scheduler = schedulers.CosineWithRestarts(
            optimizer,
            t_max=scheduling_config.get("cycle_len", nb_epochs),
            factor=scheduling_config.get('factor', 1.)
        )
    elif scheduling_config["type"] == "cosine_annealing_with_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=2, eta_min=scheduling_config.get('min_lr')
        )
    else:
        raise ValueError(f"Unknown LR scheduling type {scheduling_config}.")

    if warmup_config:
        if warmup_config.get("only_first_step", True) and task != 0:
            pass
        else:
            print("Using WarmUp")
            scheduler = schedulers.GradualWarmupScheduler(
                optimizer=optimizer, after_scheduler=scheduler, **warmup_config
            )

    return scheduler


###########################################
# More factory method that used in the Spiking or Pruning step
###########################################
def get_neuron(spiking_neuron: str):
    from spikingjelly.activation_based import neuron
    from extractor.tools.layers import LIFSpike
    spiking_neuron = spiking_neuron.lower()
    dict_neurons = {
        "if": neuron.IFNode,
        "lif": neuron.LIFNode,
        'zif-lif': LIFSpike,
    }
    if spiking_neuron not in dict_neurons:
        raise NotImplementedError(
            f"Unknown neuron type {spiking_neuron}, must be among {list(dict_neurons.keys())}"
        )
    return dict_neurons[spiking_neuron]


def get_surrogate(surrogate_function: str):
    from spikingjelly.activation_based import surrogate
    surrogate_function = surrogate_function.lower()
    dict_surrogates = {
        'none': None,
        "atan": surrogate.ATan(),
        "sig": surrogate.Sigmoid(),
        "softsign": surrogate.SoftSign(),
        "piecewise": surrogate.PiecewiseQuadratic(),
    }
    if surrogate_function not in dict_surrogates:
        raise NotImplementedError(
            f"Unknown surrogate function {dict_surrogates}, must be among {list(dict_surrogates.keys())}"
        )
    return dict_surrogates[surrogate_function]



