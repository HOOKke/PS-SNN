import argparse


def get_parser():
    # yapf: disable

    parser = argparse.ArgumentParser("IncLearner",
                                     description="Incremental Learning trainer for spiking network.")

    # Model related (concerning model and examplars):
    parser.add_argument("-m", "--model", default="spiking_oracle", type=str,
                        help="Incremental learner to train.")
    parser.add_argument("-c", "--convnet", default="spiking_rebuffi", type=str,
                        help="Backbone convnet.")
    parser.add_argument("--dropout", default=0., type=float,
                        help="Dropout value.")
    parser.add_argument("-he", "--herding", default=None, type=str,
                        help="Method to gather previous tasks' examples.")
    parser.add_argument("-mem_size_mode", type=str, default="uniform_fixed_total_mem", help="The memory distribution rule",
                        choices=["uniform_fixed_per_cls", "uniform_fixed_total_mem", "dynamic_fixed_per_cls"])
    parser.add_argument("-memory", "--memory-size", default=2000, type=int,
                        help="Max number of storable examplars.")
    # parser.add_argument("-fixed-memory", "--fixed-memory", action="store_true",
    #                     help="Instead of shrinking the memory, it's already at minimum.")
    parser.add_argument("-temp", "--temperature", default=1, type=int,
                        help="Temperature used to soften the predictions.")
    parser.add_argument("-weight_normalization", default=False, action="store_true",
                        help="Determine whether to use cos classifier (True) or use common linear classifier (False)")
    # -> and the above parameter is referred as <classifier_config> in the original code
    # ADDED
    parser.add_argument("-T", default=4, type=int,
                        help="The length of the step window of the spiking network")

    # Data related (initialization, increment...):
    parser.add_argument("-d", "--dataset", default="cifar100", type=str,
                        help="Dataset to test on.")
    parser.add_argument("-inc", "--increment", default=10, type=int,
                        help="Number of class to add per task.")
    parser.add_argument("-b", "--batch-size", default=128, type=int,
                        help="Batch size.")
    parser.add_argument("-w", "--workers", default=0, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument("-t", "--threads", default=1, type=int,
                        help="Number of threads allocated for PyTorch.")
    parser.add_argument("-v", "--validation", default=0., type=float,
                        help="Validation split (0. <= x < 1.).")
    parser.add_argument("-random", "--random-classes", action="store_true", default=False,
                        help="Randomize classes order of increment")
    parser.add_argument("-order", "--order",
                        help="List of classes ordering, to be given in options.")
    parser.add_argument("-max-task", "--max-task", default=None, type=int,
                        help="Cap the number of tasks.")
    parser.add_argument("-onehot", "--onehot", action="store_true",
                        help="Return data in onehot format.")
    parser.add_argument("-initial-increment", "--initial-increment", default=None, type=int,
                        help="Initial increment, may be bigger.")
    parser.add_argument("-sampler", "--sampler", default=None, type=str,
                        help="Elements sampler.")
    parser.add_argument("--data-path", default="./store/dataset/", type=str)
    # -> ADDED
    parser.add_argument("--opt_dir", default="./store/opt_models", type=str)
    parser.add_argument("--tensorboard_dir", default="./store/tensorboard", type=str)

    # Training related:
    parser.add_argument("-e", "--epochs", default=70, type=int,
                        help="Number of epochs per task.")
    parser.add_argument("-lr", "--lr", default=2., type=float,
                        help="Learning rate.")  # -> learning rate is default to be 2.?
    parser.add_argument("-lr-decay", "--lr-decay", default=1/5, type=float,
                        help="LR multiplied by it.")
    parser.add_argument("-opt", "--optimizer", default="sgd", type=str,
                        help="Optimizer to use.")
    parser.add_argument("-sc", "--scheduling", default=[49, 63], nargs="*", type=int,
                        help="Epoch step where to reduce the learning rate.")
    parser.add_argument("-wd", "--weight-decay", default=0.00005, type=float,
                        help="Weight decay.")
    # -> ADDED
    parser.add_argument("-spk_loss", default="ce", type=str, choices=["ce", "tet", "aug"],
                        help="type of the spiking loss")
    parser.add_argument("-val_per_n_epoch", default=10, type=int)
    parser.add_argument("-save_ckpt", action="store_false", default=True)
    parser.add_argument("-save_ft_logit", action="store_true", default=False)
    parser.add_argument('-ft_samples', type=int, default=500)

    # Misc:
    parser.add_argument("--device", default=[0], type=int, nargs="+",
                        help="GPU index to use, for cpu use -1.")
    parser.add_argument("--label", type=str,
                        help="Experience name, if used a log will be created.")
    parser.add_argument("--autolabel", action="store_true",
                        help="Auto create label based on options files.")
    parser.add_argument("-seed", "--seed", default=[1], type=int, nargs="+",
                        help="Random seed.")
    parser.add_argument("-seed-range", "--seed-range", type=int, nargs=2,
                        help="Seed range going from first number to second (both included).")
    parser.add_argument("-options", "--options", nargs="+",
                        help="A list of options files.")
    # parser.add_argument("-save", "--save-model", choices=["never", "last", "task", "first"],
    #                     default="never",
    #                     help="Save the network, either the `last` one or"
    #                          " each `task`'s ones.")
    parser.add_argument("--dump-predictions", default=False, action="store_true",
                        help="Dump the predictions and their ground-truth on disk.")
    parser.add_argument("-log", "--logging", choices=["critical", "warning", "info", "debug"],
                        default="info", help="Logging level")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to save log file. If not specified, logs will be saved to a default location based on experiment label.")
    # parser.add_argument("-resume", "--resume", default=None,
    #                     help="Resume from previously saved model, "
    #                          "must be in the format `*_task_[0-9]+\.pth`.")
    parser.add_argument("--resume-first", action="store_true", default=False)
    parser.add_argument("--recompute-meta", action="store_true", default=False)
    parser.add_argument("--no-benchmark", action="store_true", default=False)
    parser.add_argument("--detect-anomaly", action="store_true", default=False)
    # ADDED (for process control)
    parser.add_argument("-resume", "--resume", action="store_true", default=False)
    parser.add_argument("-resume_task", "--resume_task", default=0, type=int, help="The resume task id started resuming from")
    parser.add_argument("-start_task", default=0, type=int, help="The task idx from which start to save")
    parser.add_argument("-save_mem", action="store_false", default=True)
    parser.add_argument("-load_mem", action="store_true", default=False)
    # ADDED (for some parameter)
    parser.add_argument("-loss_type", default="ce", type=str)
    parser.add_argument("-analyse", action="store_true", default=False)


    parser.add_argument("--orthogonal_init_clf", action="store_true", default=False, help="Orthogonal initialization of the classifier")
    parser.add_argument("--fix_clf", action="store_true", default=False, help="Fixed classifier weight")
    

    return parser
