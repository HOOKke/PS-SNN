import copy
import json
import logging
import os
import pickle
import random
import statistics
import sys
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

import yaml
from utils import factory
from utils import logger as logger_lib
from utils import metrics, results_utils, utils

logger = logging.getLogger(__name__)


def train(args):
    ###########################################
    # The training setting initialization
    ###########################################
    # -> generate all the <options> recorded in the args
    autolabel = _set_up_options(args)
    if args["autolabel"]:
        args["label"] = autolabel

    # 生成日志文件路径
    if args.get("log_file"):
        log_file_path = args["log_file"]
    else:
        log_file_path = logger_lib.get_default_log_path(
            label=args.get("label"), 
            exp_dir=args.get("opt_dir", ".")
        )
    
    # 设置日志级别和文件路径
    logger_lib.set_logging_level(args["logging"], log_file_path)

    if args["label"]:
        logger.info(f"Label: {args['label']}")
        try:
            os.system("echo '\ek{}\e\\'".format(args["label"]))  # -> show the label in the termination
        except:
            pass
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    tenboard = SummaryWriter(os.path.join(args["tensorboard_dir"], args['label']))

    start_date = utils.get_date()

    orders = copy.deepcopy(args["order"])
    del args["order"]
    if orders is not None:
        # -> check the formation of the orders
        assert isinstance(orders, list) and len(orders)
        assert all(isinstance(o, list) for o in orders)
        assert all([isinstance(c, int) for o in orders for c in o])
    else:
        orders = [None for _ in range(len(seed_list))]

    ###########################################
    # The training process
    ###########################################
    avg_inc_accs, last_accs, forgettings = [], [], []
    # iteratively use all the seed
    for i, seed in enumerate(seed_list):
        logger.warning(f"Launch run {i+1}/{len(seed_list)}")
        args["seed"] = seed
        args["device"] = device

        start_time = time.time()

        # iteratively train the tasks
        for avg_inc_acc, last_acc, forgetting in _train(args, start_date, orders[i], i, tenboard):
            yield avg_inc_acc, last_acc, forgetting, False

        # -> it is the final result of one iteration
        avg_inc_accs.append(avg_inc_acc)
        last_accs.append(last_acc)
        forgettings.append(forgetting)

        logger.info("Training finished in {}s".format(int(time.time() - start_time)))
        yield avg_inc_acc, last_acc, forgetting, True

    tenboard.close()
    logger.info(f"Label was: {args['label']}")

    logger.info(
        "Results done on {} seeds: avg: {}, last: {}, forgetting: {}".format(
            len(seed_list), _aggregate_results(avg_inc_accs), _aggregate_results(last_accs),
            _aggregate_results(forgettings)
        )
    )
    logger.info(f"Individual results avg: {[round(100 * acc, 2) for acc in avg_inc_accs]}")
    logger.info(f"Individual results last: {[round(100 * acc, 2) for acc in last_accs]}")
    logger.info(
        f"Individual results forget: {[round(100 * fgt, 2) for fgt in forgettings]}"
    )
    logger.info(f"Command was {''.join(sys.argv)}")


# -> the whole training process of an incremental learning
def _train(args, start_date, class_order, run_id, tenboard):
    _set_global_parameters(args)  # -> set the seed and devices
    results, results_folder = _set_results(args, start_date, run_id)  # -> set the results preset and path
    args["exp_dir"] = results_folder

    inc_dataset, model = _set_data_model(args, class_order, tenboard)  # -> set the model and datasets
    metric_logger = metrics.MetricLogger(
        inc_dataset.n_tasks, inc_dataset.n_classes, inc_dataset.increments
    )  # -> the logger which log the condition of all the tasks
    #############

    DER_result = []
    for task_id in range(inc_dataset.n_tasks):
        # import pdb; pdb.set_trace()
        # if task_id >= 2:
        #     continue
        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task()
        if task_info["task"] == args["max_task"]:
            break

        model.set_task_info(task_info)

        # ----------------------
        # 1. Prepare the Task
        # ----------------------
        model.eval()
        model.before_task(task_id, train_loader)
        # -> specific model's pre-process

        # ----------------------
        # 2. Train the Task
        # ----------------------
        _train_task(args, model, train_loader, val_loader, test_loader, run_id, task_id, task_info)
        # # ImageNet part ########################################
        # _train_task(args, model, inc_dataset, train_loader, val_loader, test_loader, run_id, task_id, task_info)
        # ########################################################

        # ----------------------
        # 3. Conclude Task
        # ----------------------
        # model.eval()
        # -> the after-task operations (normally incrementally generate examplar)
        _after_task(args, model, inc_dataset, run_id, task_id, results_folder)

        # ----------------------
        # 4. Eval Task
        # ----------------------
        logger.info(f"Eval on {0}->{task_info['max_class']}")
        ypreds, ytrue = model.eval_task(test_loader)
        metric_logger.log_task(
            ypreds, ytrue, task_size=task_info["increment"], zeroshot=args.get("all_test_classes")
        )
        #####################################################
        # comparison part
        acc_stats = utils.compute_accuracy(ypreds, ytrue, increments=model._increments, n_classes=model._n_classes)
        DER_result.append(acc_stats)
        #####################################################
        # -> store the models' results
        if args["dump_predictions"] and args["label"]:
            os.makedirs(
                os.path.join(results_folder, f"prediction_{run_id}"), exist_ok=True
            )
            with open(
                os.path.join(
                    results_folder, f"predictions_{run_id}",
                    str(task_id).rjust(len(str(30)), "0") + ".pkl"
                ), "wb+"
            ) as f:
                pickle.dump((ypreds, ytrue), f)

        if args["label"]:
            logger.info(args["label"])
        logger.info(f"Avg inc acc: {round(metric_logger.last_results['incremental_accuracy'], 3)}.")
        # #############
        # logger.info(f"And in der's code the avg inc acc is: {round(utils.compute_avg_inc_acc(DER_result), 3)}.")
        # #############
        logger.info(f"Current acc: {metric_logger.last_results['accuracy']}.")
        logger.info(
            f"Avg inc acc top5: {round(metric_logger.last_results['incremental_accuracy_top5'], 3)}."
        )
        logger.info(f"Current acc top5: {metric_logger.last_results['accuracy_top5']}.")
        logger.info(f"Forgetting: {metric_logger.last_results['forgetting']}.")
        logger.info(f"Cord metric: {metric_logger.last_results['cord']}.")  # -> single class's accuracy
        if task_id > 0:
            logger.info(
                "Old accuracy: {:.2f}, mean: {:.2f}.".format(
                    metric_logger.last_results["old_accuracy"],
                    metric_logger.last_results["avg_old_accuracy"],
                )
            )
            logger.info(
                "New accuracy: {:.2f}, mean: {:.2f}.".format(
                    metric_logger.last_results["new_accuracy"],
                    metric_logger.last_results["avg_new_accuracy"]
                )
            )
        if args.get("all_test_classes"):
            logger.info(
                f"Seen classes: {metric_logger.last_results['seen_classes_accuracy']:.2f}."
            )
            logger.info(
                f"unSeen classes: {metric_logger.last_results['unseen_classes_accuracy']}."
            )

        results["results"].append(metric_logger.last_results)

        avg_inc_acc = results["results"][-1]["incremental_accuracy"]
        last_acc = results["results"][-1]["accuracy"]["total"]
        forgetting = results["results"][-1]["forgetting"]
        # Tensorboard 用法在这里
        # model._tensorboard.add_scalar(f"taskaccu/trial{run_id}", last_acc, task_id)
        yield avg_inc_acc, last_acc, forgetting
        # -> last_acc is the avg_acc of the final stage


    logger.info(
        f"Average Incremental Accuracy: {results['results'][-1]['incremental_accuracy']}."
    )
    if args["label"] is not None:
        results_utils.save_results(
            results, args["label"], args["model"], start_date, run_id, args["seed"]
        )



#############################################
# Phase's function in incremental learning
#############################################
# -> simply call the training function inside the method class... and may skip one task training according to parameters
def _train_task(config, model, train_loader, val_loader, test_loader, run_id, task_id, task_info):
    logger.info(f"Train on {task_info['min_class']}->{task_info['max_class']}")
    # -> so... how can a parameter path be a directory...
    if config["resume"] and config["save_ckpt"] and task_id < config["resume_task"]:
        model.train()
        model.load_cur_model()
        logger.info(
            f"Skipping training phase {task_id} because reloading pretrained model."
        )
    else:
        model.train()
        model.train_task(train_loader, val_loader if val_loader else test_loader)


# # ImageNet part ########################################
# def _train_task(config, model, inc_dataset, train_loader, val_loader, test_loader, run_id, task_id, task_info):
#     logger.info(f"Train on {task_info['min_class']}->{task_info['max_class']}")
#     # -> so... how can a parameter path be a directory...
#     if config["resume"] and config["save_ckpt"] and task_id < config["resume_task"]:
#         model.load_cur_model()
#         logger.info(
#             f"Skipping training phase {task_id} because reloading pretrained model."
#         )
#         inc_dataset.shared_data_inc = train_loader.dataset.share_memory
#     else:
#         model.train()
#         model.train_task(train_loader, val_loader if val_loader else test_loader)
# ########################################################


def _after_task(config, model, inc_dataset, run_id, task_id, results_folder):
    # -> same as the first condition in training period
    if config["resume"] and config["save_ckpt"] and task_id < config["resume_task"]:
        # -> load the abort examplars (maybe other setting parameter) of the method
        # model.load_after_task_info()
        model.after_task(task_id, inc_dataset)  # -> the network operation after tasks
    else:
        # -> the normal operation after the training
        model.after_task(task_id, inc_dataset)  # -> the network operation after tasks


#############################################
# Parameter setting
#############################################


# -> get the result template and the result folder path
def _set_results(config, start_date, run_id):
    if config["label"]:
        results_folder = results_utils.get_save_folder(
            config["model"], run_id, start_date, config["label"], config["seed"], path_root=config["opt_dir"])
    else:
        results_folder = results_utils.get_save_folder(
            config["model"], run_id, start_date, "default", config["seed"], path_root=config["opt_dir"])

    # -> simply get a results template prepared to store the results
    results = results_utils.get_template_results(config)

    return results, results_folder


# -> set the model and the corresponding incremental dataset
def _set_data_model(config, class_order, tenboard):
    inc_dataset = factory.get_data(config, class_order)  # -> construct the incremental dataset
    config["class_order"] = inc_dataset.class_order
    logger.info("class_order")
    logger.info(config["class_order"])

    model = factory.get_model(config, inc_dataset, tenboard)
    logger.info(f"The Model {config['model']} is initialized")

    return inc_dataset, model


# -> set the global seed and the devices settings
def _set_global_parameters(config):
    _set_seed(config["seed"], config["threads"], config["no_benchmark"], config["detect_anomaly"])
    factory.set_device(config)


# -> set the seeds for all imported packages
def _set_seed(seed, nb_threads, no_benchmark, detect_anomaly):
    logger.info(f"Set seed {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if no_benchmark:
        logger.warning("CUDA algos are not determinists but fast")  # -> introduce uncertainty
    else:
        logger.warning("CUDA algos are determinists but very slow")
    torch.backends.cudnn.deterministic = not no_benchmark  # This will slow down training.
    torch.set_num_threads(nb_threads)
    if detect_anomaly:
        # -> detect the anomalies in the autograd
        logger.info("Will detect autograd anomaly")
        torch.autograd.set_detect_anomaly(detect_anomaly)


def _set_up_options(args):
    options_path = args["options"] or []

    autolabel = []
    for option_path in options_path:
        if not os.path.exists(option_path):
            raise IOError(f"Not found options file {option_path}.")

        args.update(_parse_options(option_path))
        # -> get the file name (without the extensions i.e. .txt)
        autolabel.append(os.path.splitext(os.path.basename(option_path))[0])

    return '_'.join(autolabel)


# -> load the option information in <path>
def _parse_options(path):
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.load(f, Loader=yaml.FullLoader)
        elif path.endswith('.json'):
            return json.load(f)["config"]
        else:
            raise Exception(f"Unknown file type {path}")


#############################################
# Possible tool
#############################################


# -> sum up the list of results (including the average value and its std)
def _aggregate_results(list_results):
    res = str(round(statistics.mean(list_results) * 100, 2))
    # -> get the average results
    if len(list_results) > 1:
        res = res + ' +/- ' + str(round(statistics.stdev(list_results) * 100, 2))
    return res


