import collections

import numpy as np
import torch


class MetricLogger:
    def __init__(self, nb_tasks, nb_classes, increments):
        self.metrics = collections.defaultdict(list)

        self.nb_tasks = nb_tasks
        self.nb_classes = nb_classes
        self.increments = increments

        self._accuracy_matrix = np.ones((nb_classes, nb_tasks), dtype=np.float16) * -1
        self._task_counter = 0

    # -> log the task condition
    # -> and it will change the <_task_counter> sequentially
    def log_task(self, ypreds, ytrue, task_size, zeroshot=False):
        self.metrics["accuracy"].append(
            accuracy_per_task(ypreds, ytrue, task_size=task_size, topk=1)
        )
        self.metrics["accuracy_top5"].append(
            accuracy_per_task(ypreds, ytrue, task_size=None, topk=5)
        )
        self.metrics["accuracy_per_class"].append(
            accuracy_per_task(ypreds, ytrue, task_size=1, topk=1)
        )
        self.metrics["incremental_accuracy"].append(incremental_accuracy(self.metrics["accuracy"]))
        self.metrics["incremental_accuracy_top5"].append(
            incremental_accuracy(self.metrics["accuracy_top5"])
        )
        self.metrics["forgetting"].append(forgetting(self.metrics["accuracy"]))

        self._update_accuracy_matrix(self.metrics["accuracy_per_class"][-1])
        self.metrics["cord"].append(cord_metric(self._accuracy_matrix))

        # -> record the accuracy of the seen and unseen classes
        # -> zeroshot need to process the unseen class
        if zeroshot:
            seen_classes_indexes = np.where(ytrue < sum(self.increments[:self._task_counter+1]))[0]
            # -> indexes of the seen classes
            self.metrics["seen_classes_accuracy"].append(
                accuracy(ypreds[seen_classes_indexes], ytrue[seen_classes_indexes])
            )
            unseen_classes_indexes = np.where(
                ytrue >= sum(self.increments[:self._task_counter + 1])
            )[0]
            self.metrics["unseen_classes_accuracy"].append(
                accuracy(ypreds[unseen_classes_indexes], ytrue[unseen_classes_indexes])
            )

        if self._task_counter > 0:  # -> assure that the old task is exist
            self.metrics["old_accuracy"].append(old_accuracy(ypreds, ytrue, task_size))
            self.metrics["new_accuracy"].append(new_accuracy(ypreds, ytrue, task_size))

        self._task_counter += 1

    # -> the last task record
    @property
    def last_results(self):
        results = {
            "task_id": len(self.metrics["accuracy"]) - 1,
            "accuracy": self.metrics["accuracy"][-1],  # -> "accuracy" is including all accuracy of all tasks
            "incremental_accuracy": self.metrics["incremental_accuracy"][-1],
            "accuracy_top5": self.metrics["accuracy_top5"][-1],
            "incremental_accuracy_top5": self.metrics["incremental_accuracy_top5"][-1],
            "forgetting": self.metrics["forgetting"][-1],
            "accuracy_per_class": self.metrics["accuracy_per_class"][-1],
            "cord": self.metrics["cord"][-1]
        }

        if "old_accuracy" in self.metrics:
            results.update(
                {
                    "old_accuracy": self.metrics["old_accuracy"][-1],
                    "new_accuracy": self.metrics["new_accuracy"][-1],
                    "avg_old_accuracy": np.mean(self.metrics["old_accuracy"]),
                    "avg_new_accuracy": np.mean(self.metrics["new_accuracy"]),
                }
            )
        if "seen_classes_accuracy" in self.metrics:
            results.update(
                {
                    "seen_classes_accuracy": self.metrics["seen_classes_accuracy"],
                    "unseen_classes_accuracy": self.metrics["unseen_classes_accuracy"],
                }
            )
        return results

    # -> fixed the matrix of this metrics
    # <_accuracy_matrix> shape is [class_size, task_size]
    def _update_accuracy_matrix(self, new_accuracy_per_class):
        for k, v in new_accuracy_per_class.items():
            if k == "total":
                continue
            class_id = int(k.split("-")[0])  # -> start class id
            self._accuracy_matrix[class_id, self._task_counter] = v


# -> this function seems not to use <only> ...
# -> calculate the average accuracy of one single class up to now
def cord_metric(accuracy_matrix, only=None):
    accuracies = []

    for class_id in range(accuracy_matrix.shape[0]):
        filled_indexes = np.where(accuracy_matrix[class_id] > -1.)[0]

        # if only == "old":
        #     filled_indexes[1:]
        # elif only == "new":
        #     filled_indexes[:1]

        if len(filled_indexes) == 0:
            continue
        accuracies.append(np.mean(accuracy_matrix[class_id, filled_indexes]))
    return np.mean(accuracies).item()


# get all the increment tasks' accuracy till now
def accuracy_per_task(ypreds, ytrue, task_size=10, topk=1):
    """Computes accuracy for the whole test & per task.

    :param ypred: The predictions array.
    :param ytrue: The ground-truth array.
    :param task_size: The size of the task.
    :return: A dictionnary.
    """
    all_acc = {}

    all_acc["total"] = accuracy(ypreds, ytrue, topk=topk)

    if task_size is not None:
        for class_id in range(0, np.max(ytrue) + task_size, task_size):
            if class_id > np.max(ytrue):
                break  # -> emmm

            idxes = np.where(np.logical_and(ytrue >= class_id, ytrue < class_id + task_size))[0]

            label = "{}-{}".format(
                str(class_id).rjust(2, "0"),
                str(class_id + task_size - 1).rjust(2, "0")
            )
            all_acc[label] = accuracy(ypreds[idxes], ytrue[idxes], topk=topk)

    return all_acc


def old_accuracy(ypreds, ytrue, task_size):
    """Computes accuracy for the whole test & per task.

        :param ypred: The predictions array. including the confidence score
        :param ytrue: The ground-truth array.
        :param task_size: The size of the task.
        :return: A dictionnary.
        """
    nb_classes = ypreds.shape[1]
    old_class_indexes = np.where(ytrue < nb_classes - task_size)[0]
    return accuracy(ypreds[old_class_indexes], ytrue[old_class_indexes], topk=1)


def new_accuracy(ypreds, ytrue, task_size):
    """Computes accuracy for the whole test & per task.

    :param ypred: The predictions array.
    :param ytrue: The ground-truth array.
    :param task_size: The size of the task.
    :return: A dictionnary.
    """
    nb_classes = ypreds.shape[1]
    new_class_index = np.where(ytrue >= nb_classes - task_size)[0]
    return accuracy(ypreds[new_class_index], ytrue[new_class_index], topk=1)


# -> get the <topk> accuracy between <outputs> and <targets>
def accuracy(outputs, targets, topk=1):
    outputs, targets = torch.tensor(outputs), torch.tensor(targets)

    batch_size = targets.shape[0]
    if batch_size == 0:
        return 0.
    nb_classes = len(np.unique(targets))
    topk = min(topk, nb_classes)

    _, pred = outputs.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].reshape(-1).float().sum(0).item()
    return round(correct_k/batch_size, 3)


# -> get the average incremental accuracy of all the STAGES (not tasks)
def incremental_accuracy(accuracies):
    """Computes the average incremental accuracy as described in iCaRL.

    It is the average of the current task accuracy (tested on 0-X) with the
    previous task accuracy.

    :param acc_dict: A list TODO
    """
    return sum(task_acc["total"] for task_acc in accuracies) / len(accuracies)


# -> an evaluation measurement
# -> return the average max accuracy-decay during the learning process
def forgetting(accuracies):
    if len(accuracies) == 1:
        return 0.

    last_accuracies = accuracies[-1]
    usable_tasks = last_accuracies.keys()

    forgetting = 0.
    # -> search the max accuracy during the learning process
    for task in usable_tasks:
        if task == "total":
            continue

        max_task = 0.

        for task_accuracies in accuracies[:-1]:
            if task in task_accuracies:
                max_task = max(max_task, task_accuracies[task])

        forgetting += max_task - last_accuracies[task]

    return forgetting / len(usable_tasks)


def forward_transfer(accuracies):
    # it seems that this work finally abandon this measurement...
    pass


######################################
# Added for the performance record during the training period
######################################

class IncConfusionMeter:
    """Maintains a confusion matrix for a given classification problem.
        The ConfusionMeter constructs a confusion matrix for a multi-class
        classification problems. It does not support multi-label, multi-class problems:
        for such problems, please use MultiLabelConfusionMeter.
        Args:
            k (int): number of classes in the classification problem
            normalized (boolean): Determines whether or not the confusion matrix
                is normalized or not
    """
    def __init__(self, k, increments, normalized=False):
        self.conf = np.ndarray((k, k), dtype=np.int32)  # -> confusion matrix
        self.normalized = normalized
        self.increments = increments
        self.cum_increments = [0] + [sum(increments[:i+1]) for i in range(len(increments))]
        self.k = k

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):  # -> simply add to the source confusion block
        """Computes the confusion matrix of K x K size where K is no of classes
        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors
        """
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        # sort the formation of the predicted and the target (to a single dimension vector)
        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'
        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):  # -> seems to get a confusion block of the huge tasks rather than a single
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        conf = self.conf.astype(np.float32)
        new_conf = np.zeros([len(self.increments), len(self.increments) + 2])
        for i in range(len(self.increments)):
            idxs = range(self.cum_increments[i], self.cum_increments[i + 1])
            new_conf[i, 0] = conf[idxs, idxs].sum()
            new_conf[i, 1] = conf[self.cum_increments[i]:self.cum_increments[i + 1],
                                  self.cum_increments[i]:self.cum_increments[i + 1]].sum() - new_conf[i, 0]
            for j in range(len(self.increments)):
                new_conf[i, j + 2] = conf[self.cum_increments[i]:self.cum_increments[i + 1],
                                          self.cum_increments[j]:self.cum_increments[j + 1]].sum()
        conf = new_conf
        if self.normalized:
            return conf / conf[:, 2:].sum(1).clip(min=1e-12)[:, None]
        else:
            return conf


class ClassErrorMeter:  # Topk record
    def __init__(self, topk=[1], accuracy=False):
        super(ClassErrorMeter, self).__init__()
        self.topk = np.sort(topk)
        self.accuracy = accuracy
        self.reset()

    def reset(self):
        self.sum = {v: 0 for v in self.topk}
        self.n = 0

    def add(self, output, target):
        if isinstance(output, np.ndarray):
            output = torch.Tensor(output)
        if isinstance(target, np.ndarray):
            target = torch.Tensor(target)

        topk = self.topk
        maxk = int(topk[-1])
        no = output.shape[0]

        pred = output.topk(maxk, 1, True, True)[1]
        correct = pred == target.unsqueeze(1).repeat(1, pred.shape[1])

        for k in topk:
            self.sum[k] += no - correct[:, 0:k].sum()
        self.n += no

    def value(self, k=-1):
        if k != -1:
            assert k in self.sum.keys(), \
                'invalid k (this k was not provided at construction time)'
            if self.n == 0:
                return float('nan')
            if self.accuracy:
                return (1. - float(self.sum[k]) / self.n) * 100.0
            else:
                return float(self.sum[k]) / self.n * 100.0
        else:
            return [self.value(k_) for k_ in self.topk]


class AverageValueMeter:
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean, self.std = self.sum, np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = math.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std
