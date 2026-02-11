import datetime
import logging
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

import utils.utils
from utils.metrics import ClassErrorMeter

logger = logging.getLogger(__name__)


# -> fixed the scalar targets to a label (maybe used to calculate the distillation loss)
def to_onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def check_loss(loss):
    return not bool(torch.isnan(loss).item()) and bool((loss > 0.).item())


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d")


# -> extract the features from data in <loader>
def extract_features(model, loader):
    targets, features = [], []
    state = model.training
    model.eval()

    with torch.no_grad():
        for (_inputs, _targets) in loader:
            _targets = _targets.numpy()

            # _features = model.extract(inputs.to(model.device)).detach().cpu().numpy()
            _features = model(_inputs)['feature'].detach().cpu().numpy()

            features.append(_features)
            targets.append(_targets)

    model.train(state)

    return np.concatenate(features), np.concatenate(targets)


def extract_corr_features(model, loader):
    targets, features, corrs = [], [], []
    state = model.training
    model.eval()

    with torch.no_grad():
        for (_inputs, _targets) in loader:
            _targets = _targets.numpy()

            model_opt = model(_inputs)
            _features = model_opt['feature'].detach().cpu().numpy()
            _logit = model_opt['logit'].detach().cpu().numpy()
            _pred = np.argmax(_logit, axis=1)
            _corr = np.zeros((_features.shape[0], ))
            _corr[_pred == _targets] = 1

            features.append(_features)
            targets.append(_targets)
            corrs.append(_corr)

    model.train(state)

    return np.concatenate(features), np.concatenate(targets), np.concatenate(corrs)


def extract_logit_features_cls(model, loader, cls_id, temperature=1.0):
    targets, confs, features, corrs = [], [], [], []
    state = model.training
    model.eval()

    with torch.no_grad():
        for(_inputs, _targets) in loader:
            _targets = _targets.numpy()

            model_opt = model(_inputs)
            _features = model_opt['feature'].detach().cpu().numpy()
            _logit = model_opt['logit']
            assert _targets[0] == cls_id
            _conf = torch.softmax(_logit/temperature, dim=-1).detach().cpu().numpy()
            _conf = _conf[:, cls_id]
            _logit = _logit.detach().cpu().numpy()
            _pred = np.argmax(_logit, axis=1)
            _corr = np.zeros((_features.shape[0], ))
            _corr[_pred == _targets] = 1

            features.append(_features)
            confs.append(_conf)
            targets.append(_targets)
            corrs.append(_corr)

    model.train(state)

    return np.concatenate(features), np.concatenate(confs), np.concatenate(targets), np.concatenate(corrs)


def extract_logit_features(model, loader):
    targets, logits, features, corrs = [], [], [], []
    state = model.training
    model.eval()

    with torch.no_grad():
        for(_inputs, _targets) in loader:
            _targets = _targets.numpy()

            model_opt = model(_inputs)
            _features = model_opt['feature'].detach().cpu().numpy()
            _logit = model_opt['logit'].detach().cpu().numpy()
            _pred = np.argmax(_logit, axis=1)
            _corr = np.zeros((_features.shape[0], ))
            _corr[_pred == _targets] = 1

            features.append(_features)
            logits.append(_logit)
            targets.append(_targets)
            corrs.append(_corr)

    model.train(state)

    return np.concatenate(features), np.concatenate(logits), np.concatenate(targets), np.concatenate(corrs)


# -> compute the mean features of every category in <loader>
def compute_centroids(model, loader):
    features, targets = extract_features(model, loader)

    centroids_features, centroids_targets = [], []
    for t in np.unique(targets):
        indexes = np.where(targets == t)[0]

        centroids_features.append(np.mean(features[indexes], axis=0, keepdims=True))
        centroids_targets.append(t)

    return np.concatenate(centroids_features), np.array(centroids_targets)


# -> simply use the <model> to classify the data in <loader>
def classify(model, loader):
    targets, predictions = [], []

    for (inputs, _targets) in loader:
        outputs = model(inputs.to(model.device))
        if not isinstance(outputs, list):
            outputs = [outputs]

        preds = outputs[-1].argmax(dim=1).detach().cpu().numpy()

        predictions.append(preds)
        targets.append(_targets)

    return np.concatenate(predictions), np.concatenate(targets)


# -> take a dimensionality reduction in <embeddings> and plot them on a figure
def plot_tsne(path, embeddings, targets):
    assert embeddings.shape[0] == targets.shape[0]

    tsne = manifold.TSNE(n_components=2)

    embeddings_2d = tsne.fit_transform(embeddings)
    plt.scatter(
        embeddings_2d[..., 0],
        embeddings_2d[..., 1],
        c=targets,
        vmin=min(targets),
        vmax=max(targets),
        s=10,
        cmap=mpl.cm.get_cmap('RdYlBu')
    )

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(path)


# -> append weights for classifier part in the whole model
def add_new_weights(network, weight_generation, current_nb_classes, task_size, inc_dataset):
    if isinstance(weight_generation, str):
        warnings.warn("Use a dict for weight_generation instead of str", DeprecationWarning)
        weight_generation = {"type": weight_generation}

    # -> generate weight by the class embeddings (specially for cosine classifier)
    if weight_generation["type"] == "imprinted":
        logger.info("Generating imprinted weights")

        # -> this function is used to append new weight in the cosine classifier (use the embeddings of former class)
        network.add_imprinted_classes(
            list(range(current_nb_classes, current_nb_classes + task_size)), inc_dataset,
            **weight_generation
        )

    elif weight_generation["type"] == "embedding":
        logger.info("Generate embedding weights")

        mean_embeddings = []
        for class_index in range(current_nb_classes, current_nb_classes + task_size):
            _, loader = inc_dataset.get_custom_loader([class_index])
            features, _ = extract_features(network, loader)
            features = features / np.linalg.norm(features, axis=-1)[..., None]

            mean = np.mean(features, axis=0)
            if weight_generation.get("proxy_per_class", 1) == 1:
                mean_embeddings.append(mean)
            else:
                std = np.std(features, axis=0, ddof=1)
                mean_embeddings.extend(
                    [
                        np.random.normal(loc=mean, scale=std)
                        for _ in range(weight_generation.get("proxy_per_class", 1))
                    ]
                )  # -> it shows only a little difference with the imprinted one

        network.add_custom_weights(np.stack(mean_embeddings))

    elif weight_generation["type"] == "basic":
        network.add_classes(task_size)
    elif weight_generation["type"] == "ghosts":
        # -> add weights base on the artificial category and items?
        features, targets = weight_generation["ghosts"]
        features = features.cpu().numpy()
        targets = targets.cpu().numpy()

        weights = []
        for class_id in range(current_nb_classes, current_nb_classes + task_size):
            indexes = np.where(targets == class_id)[0]

            class_features = features[indexes]
            if len(class_features) == 0:
                raise Exception(f"No ghost class_id={class_id} for weight generation!")
            weights.append(np.mean(class_features, axis=0))

        weights = torch.tensor(np.stack(weights)).float()
        network.add_custom_weights(weights, ponderate=weight_generation.get("ponderate"))
    else:
        raise ValueError(f"Unknown weight generation type {weight_generation['type']}")


# -> clustering the embeddings of given features and related the result with the labels
def apply_kmeans(features, targets, nb_clusters, pre_normalization):
    logger.info(
        f"Kmeans on {len(features)} samples (pre-normalized: {[pre_normalization]} "
        f"with {nb_clusters} clusters per class)"
    )

    new_features = []
    new_targets = []
    for class_index in np.unique(targets):
        # -> KMeans is to cluster the groups of features
        kmeans = KMeans(n_clusters=nb_clusters)

        class_sample_indexes = np.where(targets == class_index)[0]
        class_features = features[class_sample_indexes]
        class_targets = np.ones((nb_clusters,)) * class_index

        if pre_normalization:
            class_features = class_features / np.linalg.norm(class_features, axis=-1).reshape(-1, 1)

        kmeans.fit(class_features)
        new_features.append(kmeans.cluster_centers_)
        new_targets.append(class_targets)

    return np.concatenate(new_features), np.concatenate(new_targets)


# -> KNN algorithm is used to classify the test samples according to its proximity to the training sample
# -> This function is just used to test the result of KNN?
def apply_knn(
    features,
    targets,
    features_test,
    targets_test,
    nb_neighbors,
    normalize=True,
    weights="uniform",
):
    logger.info(
        f"KNN with {nb_neighbors} and pre-normalized features: {normalize}, weights: {weights}"
    )

    if normalize:
        features = features / np.linalg.norm(features, axis=-1).reshape(-1, 1)

    knn = KNeighborsClassifier(n_neighbors=nb_neighbors, n_jobs=10, weights=weights)
    knn.fit(features, targets)

    if normalize:
        features_test = features_test / np.linalg.norm(features_test, axis=-1).reshape(-1, 1)

    pred_targets = knn.predict(features_test)

    return pred_targets, targets_test


# -> select specific samples
def select_class_samples(samples, targets, selected_class):
    indexes = np.where(targets == selected_class)[0]
    return samples[indexes], targets[indexes]


# -> ... what's the meaning?
def matrix_infinity_norm(matrix):
    matrix = torch.abs(matrix)

    summed_col = matrix.sum(1)  # Shape (w, )
    return torch.max(summed_col)


def grad_cam(spatial_features, selected_logits):
    batch_size = spatial_features.shape[0]
    assert batch_size == len(selected_logits)

    formated_logits = [selected_logits[i] for i in range(batch_size)]

    import pdb
    pdb.set_trace()
    grads = torch.autograd.grad(
        formated_logits, spatial_features, retain_graph=True, create_graph=True
    )

    assert grads.shape == spatial_features.shape

    return grads


def get_featnorm_grouped_by_class(network, cur_n_cls, loader, device=None):
    """
    Ret: feat_norms: list of list
            feat_norms[idx] is the list of feature norm of the images for class idx.
    """
    assert device is not None
    feats = [[] for i in range(cur_n_cls)]
    feat_norms = np.zeros(cur_n_cls)
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feat = network(x)['feature'].cpu()
            # for i, lbl in enumerate(y):
            #     feats[lbl].append(feat[y == lbl])
            for lbl in range(cur_n_cls):
                feats[lbl].append(feat[y == lbl])
    for i in range(len(feats)):
        if len(feats[i]) != 0:
            feat_cls = torch.cat((feats[i]))
            feat_norms[i] = torch.norm(feat_cls, p=2, dim=1).mean().data.numpy()
    return feat_norms


def display_weight_norm(logger, network, increments, tag):
    weight_norms = [[] for _ in range(len(increments))]
    increments = np.cumsum(np.array(increments))
    for idx in range(network.module.classifier.weight.shape[0]):
        norm = torch.norm(network.module.classifier.weight[idx].data, p=2).item()
        for i in range(len(weight_norms)):
            if idx < increments[i]:
                break
        weight_norms[i].append(round(norm, 3))
    avg_weight_norm = []
    for idx in range(len(weight_norms)):
        avg_weight_norm.append(round(np.array(weight_norms[idx]).mean(), 3))
    logger.info("%s: Weight norm per class %s" % (tag, str(avg_weight_norm)))


def display_feature_norm(logger, network, loader, n_classes, increments, tag, return_norm=False, device=None):
    avg_feat_norm_per_cls = get_featnorm_grouped_by_class(network, n_classes, loader, device)
    feature_norms = [[] for _ in range(len(increments))]
    increments = np.cumsum(np.array(increments))
    for idx in range(len(avg_feat_norm_per_cls)):
        for i in range(len(feature_norms)):
            if idx < increments[i]:  #Find the mapping from class idx to step i.
                break
        feature_norms[i].append(round(avg_feat_norm_per_cls[idx], 3))
    avg_feature_norm = []
    for idx in range(len(feature_norms)):
        avg_feature_norm.append(round(np.array(feature_norms[idx]).mean(), 3))
    logger.info("%s: Feature norm per class %s" % (tag, str(avg_feature_norm)))
    if return_norm:
        return avg_feature_norm
    else:
        return


#############################################
# -> Appended for the DER
#############################################
def update_classes_mean(network, inc_dataset, n_classes, task_size, share_memory=None, metric="cosine", EPSILON=1e-8):
    loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                     inc_dataset.targets_inc,
                                     shuffle=False,
                                     share_memory=share_memory,
                                     mode="test")
    class_means = np.zeros((n_classes, network.module.features_dim))
    count = np.zeros(n_classes)
    network.eval()
    with torch.no_grad():
        for input_dict in loader:
            x, y = input_dict["inputs"], input_dict["targets"]
            feat = network(x.cuda())['feature']
            for lbl in torch.unique(y):
                class_means[lbl] += feat[y == lbl].sum(0).cpu().numpy()
                count[lbl] += feat[y == lbl].shape[0]
        for i in range(n_classes):
            class_means[i] /= count[i]
            if metric == "cosine" or metric == "weight":
                class_means[i] /= (np.linalg.norm(class_means) + EPSILON)
    return class_means


def compute_accuracy(ypred, ytrue, increments, n_classes):
    all_acc = {"top1": {}, "top5": {}}
    topk = 5 if n_classes >= 5 else n_classes
    ncls = np.unique(ytrue).shape[0]
    if topk > ncls:
        topk = ncls
    all_acc_meter = ClassErrorMeter(topk=[1, topk], accuracy=True)
    all_acc_meter.add(ypred, ytrue)
    all_acc["top1"]["total"] = round(all_acc_meter.value()[0], 3)
    all_acc["top5"]["total"] = round(all_acc_meter.value()[1], 3)
    # all_acc["total"] = round((ypred == ytrue).sum() / len(ytrue), 3)

    # for class_id in range(0, np.max(ytrue), task_size):
    start, end = 0, 0
    for i in range(len(increments)):
        if increments[i] <= 0:
            pass
        else:
            start = end
            end += increments[i]
            if start < end:
                label = "{}-{}".format(str(start).rjust(2, "0"), str(end - 1).rjust(2, "0"))
            else:
                label = "{}-{}".format(str(start).rjust(2, "0"), str(end).rjust(2, "0"))

            idxes = np.where(np.logical_and(ytrue >= start, ytrue < end))[0]
            if idxes.shape[0] == 0:
                all_acc["top1"][label] = 0.
                all_acc["top5"][label] = 0.
                continue
            topk_ = 5 if increments[i] >= 5 else increments[i]
            ncls = np.unique(ytrue[idxes]).shape[0]
            if topk_ > ncls:
                topk_ = ncls
            cur_acc_meter = ClassErrorMeter(topk=[1, topk_], accuracy=True)
            cur_acc_meter.add(ypred[idxes], ytrue[idxes])
            top1_acc = (ypred[idxes].argmax(1) == ytrue[idxes]).sum() / idxes.shape[0] * 100
            all_acc["top1"][label] = round(top1_acc, 3)
            all_acc["top5"][label] = round(cur_acc_meter.value()[1], 3)
            # all_acc[label] = round((ypred[idxes] == ytrue[idxes]).sum() / len(idxes), 3)

    return all_acc


def compute_avg_inc_acc(results):
    """Computes the average incremental accuracy as defined in iCaRL.

    The average incremental accuracies at task X are the average of accuracies
    at task 0, 1, ..., and X.

    :param accs: A list of dict for per-class accuracy at each step.
    :return: A float.
    """
    top1_tasks_accuracy = [r['top1']["total"] for r in results]
    top1acc = sum(top1_tasks_accuracy) / len(top1_tasks_accuracy)
    return top1acc


def TET_loss(outputs, labels, means, lamb, temp=1.0, reduction='mean'):
    assert len(outputs.shape) == 3  # -> should be [B, T, N]
    Loss_es = 0
    # T = outputs.shape[0]
    T = outputs.shape[1]
    for t in range(T):
        # Loss_es += F.cross_entropy(outputs[t], labels, reduction=reduction)
        # Loss_es += F.cross_entropy(outputs[t], labels)
        Loss_es += F.cross_entropy(outputs[:, t, ...]/temp, labels, reduction=reduction)
    Loss_es = Loss_es / T
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss(reduction=reduction)
        # MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)
        if reduction == "none":
            Loss_mmd = torch.mean(torch.sum(Loss_mmd, dim=-1), dim=-1)
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd


def TET_loss_auged(auged_opts, src_opts, labels, means, lamb, reduction='mean'):
    assert len(auged_opts.shape) == 3  # -> should be [B, T, N]
    Loss_es = 0
    T = auged_opts.shape[1]
    for t in range(T):
        Loss_es += F.cross_entropy(auged_opts[:, t, ...], labels, reduction=reduction)
    Loss_es = Loss_es / T
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss(reduction=reduction)
        y = torch.zeros_like(src_opts).fill_(means)
        Loss_mmd = MMDLoss(src_opts, y)
        if reduction == "none":
            Loss_mmd = torch.mean(torch.sum(Loss_mmd, dim=-1), dim=-1)
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd


#######################################################
# Appended for my own method
#######################################################
########################################################
# customed finetune function which used by my own
def finetune_clf_step(
    loader,
    network,
    optimizer,
    scheduler,
    device,
    temperature=1.0,
    nepoch=30,
    spiking=False
):
    network.eval()
    logger.info("Begin finetuning last layer")

    for i in range(nepoch):
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)['logit']
            if spiking:
                outputs = torch.mean(outputs, dim=1)  # [B, T, N] -> [B, N]
            _, preds = outputs.max(1)
            optimizer.zero_grad()

            loss_ce = F.cross_entropy(outputs / temperature, targets)
            loss = loss_ce

            loss.backward()
            optimizer.step()
            total_loss += loss * inputs.size(0)
            total_correct += (preds == targets).sum()
            total_count += inputs.size(0)

        scheduler.step()
        logger.info("Epoch %d finetuning loss %.3f acc %.3f" %
                   (i, total_loss.item() / total_count, total_correct.item() / total_count))


def custom_finetune_clf(
    inc_dataset,
    logger,
    network,
    n_class,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    temperature=5.0,
    device=None,
    refresh_param=True,  # -> whether to refresh the parameter of the clf
    loader_type="balanced",  # -> which dataset to be used
):
    assert device is not None

    ################################
    # Change the finetune dataset
    if loader_type == 'balanced':
        loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="balanced_train")
    elif loader_type == 'old':
        loader = inc_dataset._get_loader(inc_dataset.data_memory, inc_dataset.targets_memory, mode="train")
    elif loader_type == 'icarl':
        filtered_x, filtered_y = construct_designed_balanced_subset(inc_dataset, network)
        loader = inc_dataset._get_loader(filtered_x, filtered_y, mode="train")
    elif loader_type == 'fullset':
        loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="train")
    elif loader_type == 'all':
        _, _, loader = inc_dataset.get_custom_loader(
            list(range(n_class)), mode="train",
        )
    else:
        raise NotImplementedError(f"<{loader_type}> is not a valid loader type in the finetune step")
    logger.info(f"Loader type in finetuning step is {loader_type}.")
    ################################

    ######################################################
    # RESET CLF
    if refresh_param:
        network.module.classifier.reset_parameters()
        logger.info("Reset the clf parameter")

    optim = SGD(network.module.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)
    finetune_clf_step(
        loader=loader, network=network, optimizer=optim, scheduler=scheduler,
        device=device, temperature=temperature, nepoch=nepoch
    )
    return network


def finetune_clf_step_logits_ctrl(
    loader,
    network,
    optimizer,
    scheduler,
    device,
    old_cls_num,
    ctrl_st=15,
    temperature=1.0,
    nepoch=30
):
    network.eval()
    logger.info("Begin finetuning last layer")

    effect_fac = 8
    change_fac = 4

    logit_align = 0.
    last_aug = False  # -> what measurement does last step do
    st_diff = 0.
    last_diff = 0.
    max_change = 1000.0

    for i in range(nepoch):
        total_loss = 0.0
        total_correct = 0.0
        old_cnt, new_cnt = 0, 0
        old_summ, new_summ = 0., 0.

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = network(inputs)['logit']  # -> [B, T, N]

            freq_opts = torch.mean(outputs, dim=1)
            _, preds = freq_opts.max(1)

            old_classes = targets < old_cls_num
            new_classes = targets >= old_cls_num
            ################################
            # The part to collect the logit information
            ################################
            if torch.sum(old_classes) > 0:
                old_outputs, old_targets = freq_opts[old_classes], targets[old_classes]
                old_corr = old_outputs[torch.arange(old_outputs.shape[0]), old_targets]
                assert len(old_corr.shape) == 1
                old_summ += torch.sum(old_corr).item()
            if torch.sum(new_classes) > 0:
                new_outputs, new_targets = freq_opts[new_classes], targets[new_classes]
                new_corr = new_outputs[torch.arange(new_outputs.shape[0]), new_targets]
                assert len(new_corr.shape) == 1
                new_summ += torch.sum(new_corr).item()

            if i >= ctrl_st:
                aug_mtx = torch.zeros_like(outputs).to(outputs.device)
                aug_mtx[..., old_cls_num:] = logit_align
                outputs = outputs + aug_mtx

            optimizer.zero_grad()

            freq_opts = torch.mean(outputs, dim=1)  # -> [B, N]
            loss_ce = F.cross_entropy(freq_opts / temperature, targets)

            loss = loss_ce

            loss.backward()
            optimizer.step()
            total_loss += loss * inputs.size(0)
            total_correct += (preds == targets).sum()
            old_cnt += torch.sum(old_classes).to(torch.int).item()
            new_cnt += torch.sum(new_classes).to(torch.int).item()

            ################################
            # The part to adjust the classifier's weight distribution
            ################################
            clf_w = network.module.classifier.weight.data
            cls_dim, ft_dim = clf_w.shape[0], clf_w.shape[1]

        scheduler.step()
        avg_old = old_summ / old_cnt
        avg_new = new_summ / new_cnt
        if i >= ctrl_st:
            if avg_new - avg_old > 0:
                diff = avg_new - avg_old
                if not last_aug:
                    logit_align += diff
                    last_aug = True
                    max_change = 0.
                    st_diff = diff
                else:
                    if st_diff - diff > st_diff / effect_fac and last_diff - diff < max_change / change_fac:
                        logit_align += diff
                        max_change = 0.
                        st_diff = diff
                    elif st_diff < diff:
                        logit_align += diff - st_diff
                        st_diff = diff
                    else:
                        max_change = max(last_diff - diff, max_change)
            else:
                diff = avg_old - avg_new
                if last_aug:
                    logit_align -= diff
                    last_aug = False
                    max_change = 0.
                    st_diff = diff
                else:
                    if st_diff - diff > st_diff / effect_fac and last_diff - diff < max_change / change_fac:
                        logit_align -= diff
                        max_change = 0.
                        st_diff = diff
                    elif st_diff < diff:
                        logit_align -= diff - st_diff
                        st_diff = diff
                    else:
                        max_change = max(last_diff - diff, max_change)

            last_diff = diff
            logger.info(f"logit_align: {round(logit_align, 2)}")
        total_count = old_cnt + new_cnt
        logger.info("Epoch %d finetuning loss %.3f acc %.3f, new avg %.2f, old avg %.2f" %
                    (i, total_loss.item() / total_count, total_correct.item() / total_count,
                     avg_new, avg_old))


def finetune_spiking_classifier(
    inc_dataset,
    logger,
    network,
    n_class,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    temperature=5.0,
    device=None,
    new_cls_size=0,
    refresh_param=True,  # -> whether to refresh the parameter of the clf
    loader_type="balanced",  # -> which dataset to be used
    logit_ctrl=False,
):
    assert device is not None

    ######################################################
    # RESET CLF
    if refresh_param:
        network.module.classifier.reset_parameters()
        logger.info("Reset the clf parameter")
    ######################################################
    # Logit control
    if logit_ctrl:
        logger.info(f"We use logit control to make the result balance, and temperature is {temperature}...")

    ################################
    # Change the finetune dataset
    if loader_type == 'balanced':
        loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="balanced_train")
    elif loader_type == 'old':
        loader = inc_dataset._get_loader(inc_dataset.data_memory, inc_dataset.targets_memory, mode="train")
    elif loader_type == 'icarl':
        filtered_x, filtered_y = construct_designed_balanced_subset(inc_dataset, network)
        loader = inc_dataset._get_loader(filtered_x, filtered_y, mode="train")
    elif loader_type == 'fullset':
        loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="train")
    elif loader_type == 'all':
        _, _, loader = inc_dataset.get_custom_loader(
            list(range(n_class)), mode="train",
        )
    else:
        raise NotImplementedError(f"<{loader_type}> is not a valid loader type in the finetune step")
    logger.info(f"Loader type in finetuning step is {loader_type}.")
    ################################

    network.eval()
    logger.info("SHOULD the finetune step of SPIKING CLF use SGD and MULTI-STEP remain to be a DOUBT...")
    optim = SGD(network.module.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)
    if logit_ctrl:
        finetune_clf_step_logits_ctrl(
            loader=loader, network=network, optimizer=optim, scheduler=scheduler, device=device,
            old_cls_num=n_class-new_cls_size, ctrl_st=scheduling[-1],
            temperature=temperature, nepoch=nepoch
        )
    else:
        finetune_clf_step(
            loader=loader, network=network, optimizer=optim, scheduler=scheduler,
            device=device, temperature=temperature, nepoch=nepoch, spiking=True
        )

    return network


def finetune_logit_ctrl_simpled(
    loader,
    network,
    optimizer,
    scheduler,
    device,
    old_cls_num,
    ctrl_st=10,
    nepoch=30,
    v_const=0.01,
    b_const=0.01,
    spiking=False
):
    network.eval()
    logger.info(f"Begin finetuning last layer, old_task is {old_cls_num}")

    logit_align = 0.
    last_diff = 0.
    append_diff = 0.

    for i in range(nepoch):
        total_loss, total_correct = 0.0, 0.0
        old_cnt, new_cnt = 0, 0
        old_summ, new_summ = 0., 0.

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = network(inputs)['logit']
            if spiking:
                outputs = torch.mean(outputs, dim=1) # [B, T, N] -> [B, N]
            _, preds = outputs.max(1)

            old_classes = targets < old_cls_num
            new_classes = targets >= old_cls_num
            ################################
            # The part to collect the logit information
            ################################
            # if torch.sum(old_classes) > 0:
            #     old_outputs, old_targets = outputs[old_classes], targets[old_classes]
            #     old_corr = old_outputs[torch.arange(old_outputs.shape[0]), old_targets]
            #     assert len(old_corr.shape) == 1
            #     old_summ += torch.sum(old_corr).item()
            # if torch.sum(new_classes) > 0:
            #     new_outputs, new_targets = outputs[new_classes], targets[new_classes]
            #     new_corr = new_outputs[torch.arange(new_outputs.shape[0]), new_targets]
            #     assert len(new_corr.shape) == 1
            #     new_summ += torch.sum(new_corr).item()

            old_corr = outputs[torch.arange(outputs.shape[0])[old_classes], targets[old_classes]]
            old_summ += torch.sum(old_corr).item()
            new_corr = outputs[torch.arange(outputs.shape[0])[new_classes], targets[new_classes]]
            new_summ += torch.sum(new_corr).item()

            aug_mtx = torch.zeros_like(outputs).to(outputs.device)
            if i >= ctrl_st:
                # append the logit-correction in all old and new logit
                aug_mtx[..., old_cls_num:] = logit_align

                outputs = outputs + aug_mtx

            optimizer.zero_grad()

            loss_ce = F.cross_entropy(outputs + aug_mtx, targets)

            loss_ce.backward()
            optimizer.step()
            total_loss += loss_ce * inputs.size(0)
            total_correct += (preds == targets).sum()
            old_cnt += torch.sum(old_classes).to(torch.int).item()
            new_cnt += torch.sum(new_classes).to(torch.int).item()

        scheduler.step()
        if i >= ctrl_st and i % 1 == 0:
            avg_old = old_summ / old_cnt
            avg_new = new_summ / new_cnt
            diff = (avg_new - avg_old) / 2
            if logit_align == 0. or \
                    (abs(last_diff - diff) < v_const and abs(diff) > b_const):
                    # (abs(last_diff - diff) < v_const and abs(diff) > b_const and abs((diff-append_diff)/append_diff) > 1/8):
                logit_align += diff
                append_diff = diff

            last_diff = diff
            logger.info(f"logit_align: {logit_align:.3f}, diff: {diff:.3f}, avg_old: {avg_old:.3f}, avg_new: {avg_new:.3f}")
        total_count = old_cnt + new_cnt
        logger.info(
            "Epoch %d finetuning loss %.3f acc %.3f" %
            (i, total_loss.item() / total_count, total_correct.item() / total_count)
        )


def finetune_clf_fixed(
    inc_dataset,
    logger,
    network,
    n_class,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    temperature=5.0,
    device=None,
    new_cls_size=0,
    refresh_param=True,  # -> whether to refresh the parameter of the clf
    loader_type="balanced",  # -> which dataset to be used
    logit_ctrl=False,
    spiking=False,
):
    assert device is not None

    ######################################################
    # RESET CLF
    if refresh_param:
        network.module.classifier.reset_parameters()
        logger.info("Reset the clf parameter")
    ######################################################
    # Logit control
    if logit_ctrl:
        logger.info(f"We use logit control to make the result balance, and temperature is {temperature}...")

    ################################
    # Change the finetune dataset
    if loader_type == 'balanced':
        loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="balanced_train")
    elif loader_type == 'old':
        loader = inc_dataset._get_loader(inc_dataset.data_memory, inc_dataset.targets_memory, mode="train")
    elif loader_type == 'icarl':
        filtered_x, filtered_y = construct_designed_balanced_subset(inc_dataset, network)
        loader = inc_dataset._get_loader(filtered_x, filtered_y, mode="train")
    elif loader_type == 'fullset':
        loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="train")
    elif loader_type == 'all':
        _, _, loader = inc_dataset.get_custom_loader(
            list(range(n_class)), mode="train",
        )
    else:
        raise NotImplementedError(f"<{loader_type}> is not a valid loader type in the finetune step")
    logger.info(f"Loader type in finetuning step is {loader_type}. nb {len(loader.dataset)}")
    ################################

    network.eval()
    logger.info("SHOULD the finetune step of SPIKING CLF use SGD and MULTI-STEP remain to be a DOUBT...")
    if hasattr(network.module, 'finetune_parameters'):
        params = network.module.finetune_parameters()
    else:
        params = network.module.classifier.parameters()
    optim = SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)
    if logit_ctrl:
        finetune_logit_ctrl_simpled(
            loader=loader, network=network, optimizer=optim, scheduler=scheduler, device=device,
            old_cls_num=n_class-new_cls_size, ctrl_st=15, nepoch=nepoch, spiking=spiking,
        )
    else:
        finetune_clf_step(
            loader=loader, network=network, optimizer=optim, scheduler=scheduler,
            device=device, temperature=temperature, nepoch=nepoch, spiking=spiking
        )

    return network


def binary_weight_check(line_weight, out_dim):  # -> check the weight distribution in binary task
    old_weight = line_weight[:-out_dim]
    new_weight = line_weight[-out_dim:]
    old_abs_magn = torch.mean(torch.abs(old_weight)).item()
    new_abs_magn = torch.mean(torch.abs(new_weight)).item()
    old_neg_magn = torch.mean(old_weight[old_weight < 0]).item()
    old_pos_magn = torch.mean(old_weight[old_weight > 0]).item()
    new_neg_magn = torch.mean(new_weight[new_weight < 0]).item()
    new_pos_magn = torch.mean(new_weight[new_weight > 0]).item()
    logger.info(f"old_abs_magnitude: {old_abs_magn: .3f}, new_abs_magnitude: {new_abs_magn: .3f}")
    logger.info(f"old_neg_magnitude: {old_neg_magn: .3f}, old_pos_magnitude: {old_pos_magn: .3f}")
    logger.info(f"new_neg_magnitude: {new_neg_magn: .3f}, new_pos_magnitude: {new_pos_magn: .3f}")

    neg_weight = line_weight[line_weight < 0]
    pos_weight = line_weight[line_weight > 0]
    neg_magn = -torch.mean(torch.abs(neg_weight)).item()
    pos_magn = torch.mean(torch.abs(pos_weight)).item()
    logger.info(f"all_neg_magnitude: {neg_magn: .3f}, all_pos_magnitude: {pos_magn: .3f}")


def binary_finetune(
    inc_dataset,
    logger,
    network,
    n_class,
    task_size,
    taski,
    save_dir,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    device=None,
):
    from sklearn.metrics import roc_auc_score
    assert device is not None
    assert save_dir is not None
    _, _, train_loader = inc_dataset.get_custom_loader(
        list(range(n_class)), data_source="train", mode="train"
    )
    _, _, test_loader = inc_dataset.get_custom_loader(
        list(range(n_class)), data_source="test", mode="test"
    )
    logger.info(f"test-dataset nb {len(test_loader.dataset)}")

    network.module.classifier.reset_parameters()
    optim = SGD(network.module.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)

    logger.info("Try to get the best ability of current performance")
    network.eval()
    loss_func = nn.BCEWithLogitsLoss()
    for i in range(nepoch):
        total_loss, total_correct, total_cnt = 0.0, 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)['logit']
            # change the targets
            old_classes = targets < (n_class - task_size)
            new_classes = targets >= (n_class - task_size)
            targets[old_classes] = 0
            targets[new_classes] = 1

            preds = (outputs[:, 0] > 0.5).float()

            optim.zero_grad()

            loss = loss_func(outputs[:, 0], targets.float())

            loss.backward()
            optim.step()
            total_loss += loss * inputs.size(0)
            total_correct += (preds == targets).sum()
            total_cnt += inputs.size(0)

        scheduler.step()
        logger.info(f"Epoch {i} loss {total_loss.item()/total_cnt: .3f}, acc {total_correct.item()/total_cnt: .3f}")
    logger.info(f"Finish training in binary problem")

    eval_cnt, eval_corr = 0, 0
    logits = [[] for _ in range(2)]
    eval_score = []
    eval_lbls = []
    network.eval()
    with torch.no_grad():
        for i, (inputs, lbls) in enumerate(test_loader):
            inputs = inputs.to(device)
            lbls = lbls.to(device)
            outputs = network(inputs)['logit']
            preds = (outputs[:, 0] > 0.5).float()

            # change the targets
            old_classes = lbls < (n_class - task_size)
            new_classes = lbls >= (n_class - task_size)
            lbls[old_classes] = 0
            lbls[new_classes] = 1

            eval_cnt += inputs.size(0)
            eval_corr += (preds == lbls).sum()
            eval_score.append(outputs[:, 0].cpu())
            eval_lbls.append(lbls.cpu())

            for label in range(2):
                logits[label].append(outputs.cpu()[lbls == label][:, 0])

    eval_score = torch.cat((eval_score)).numpy()
    eval_lbls = torch.cat((eval_lbls)).numpy()
    eval_auc = roc_auc_score(eval_lbls, eval_score)
    logger.info(f"Finish the evaluation, and the final check results is {eval_corr.item()/eval_cnt: .3f}")
    logger.info(f"And the binary auc score is {eval_auc}")
    for label in range(2):
        logits[label] = torch.cat((logits[label])).numpy()
    saved_dict = {
        'logit': logits
    }
    logit_save_pth = os.path.join(save_dir, f"step{taski}_binary.npy")
    np.save(logit_save_pth, saved_dict, allow_pickle=True)
    logger.info(f"Save binary-task result at path {logit_save_pth}")


##########################
# finetune by selected samples
def icarl_filtering(x, y, cls_id, minsize, inc_dataset, net):
    from utils.network.memory import select_examplars
    assert y.shape[0] > minsize
    for i in range(y.shape[0]):
        assert y[i] == cls_id

    loader = inc_dataset._get_loader(x, y, shuffle=False, mode='test')
    features, _ = extract_features(net, loader)
    _, idxes = select_examplars(features, minsize)
    assert len(idxes) == minsize
    return idxes


def construct_designed_balanced_subset(inc_dataset, net):
    xdata, ydata = [], []
    x, y = inc_dataset.data_inc, inc_dataset.targets_inc
    minsize = np.inf
    for cls_ in np.unique(y):
        xdata.append(x[y == cls_])
        ydata.append(y[y == cls_])
        if ydata[-1].shape[0] < minsize:
            minsize = ydata[-1].shape[0]
    for i in range(len(xdata)):
        assert xdata[i].shape[0] >= minsize
        if xdata[i].shape[0] > minsize:
            idxes = icarl_filtering(xdata[i], ydata[i], ydata[i][0], minsize, inc_dataset, net)
            assert len(idxes) == minsize
            xdata[i] = xdata[i][idxes]
            ydata[i] = ydata[i][idxes]

    return np.concatenate(xdata, 0), np.concatenate(ydata, 0)


##########################
# The part concerning about plotting and analysing (of feature distribution)
##########################
# -> generate logit of different part of representations
def get_split_corres_logit_by_class(network, cur_n_cls, loader, device=None):
    assert device is not None
    fts = [[] for i in range(cur_n_cls)]
    old_dim = network.last_dim
    new_dim = network.features_dim - old_dim
    logits = [[] for i in range(cur_n_cls)]
    old_logits = [[] for i in range(cur_n_cls)]
    new_logits = [[] for i in range(cur_n_cls)]

    network.eval()
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device)
            opt_dict = network.split_forward(x)
            for lbl in range(cur_n_cls):
                fts[lbl].append(opt_dict['feature'].cpu()[y == lbl])
                logits[lbl].append(opt_dict['logit'].cpu()[y == lbl])
                old_logits[lbl].append(opt_dict['old_logit'].cpu()[y == lbl])
                new_logits[lbl].append(opt_dict['new_logit'].cpu()[y == lbl])
    for i in range(len(fts)):
        fts[i] = torch.cat((fts[i])).numpy()
        logits[i] = torch.cat((logits[i])).numpy()
        old_logits[i] = torch.cat((old_logits[i])).numpy()
        new_logits[i] = torch.cat((new_logits[i])).numpy()

    return fts, old_dim, new_dim, logits, old_logits, new_logits


def get_split_feature_logit(
    logger, eb_network, cur_n_cls, loader,
    memory_cls=None, memory_loader=None,
    device=None, save_dir=None, taski=0
):
    fts, old_dim, new_dim, logits, old_logits, new_logits = \
        get_split_ft_logit_by_class(eb_network, cur_n_cls, loader, device)
    saved_dict = {
        'feature': fts,
        'old_dim': old_dim,
        'new_dim': new_dim,
        'logit': logits,
        'old_logit': old_logits,
        'new_logit': new_logits
    }
    if memory_loader is not None:
        assert memory_cls is not None
        mem_fts, _, _, mem_logits, o_mem_logits, n_mem_logits = \
            get_split_ft_logit_by_class(eb_network, memory_cls, memory_loader, device)
        saved_dict['mem_feature'] = mem_fts
        saved_dict['mem_logit'] = mem_logits
        saved_dict['old_mem_logit'] = o_mem_logits
        saved_dict['new_mem_logit'] = n_mem_logits
    if save_dir is not None:
        ft_save_pth = os.path.join(save_dir, f"step{taski}_ft.npy")
        np.save(ft_save_pth, saved_dict, allow_pickle=True)
        logger.info(f"Save feature in the {ft_save_pth}")


# -> generate feature and logit for each steps
def get_split_ft_logit_by_class(network, cur_n_cls, loader, device=None):
    assert device is not None
    fts = [[] for i in range(cur_n_cls)]
    logits = [[] for i in range(cur_n_cls)]

    network.eval()
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device)
            opt_dict = network(x)
            for lbl in range(cur_n_cls):
                fts[lbl].append(opt_dict['feature'].cpu()[y == lbl])
                logits[lbl].append(opt_dict['logit'].cpu()[y == lbl])
    for i in range(len(fts)):
        fts[i] = torch.cat((fts[i])).numpy()
        logits[i] = torch.cat((logits[i])).numpy()

    saved_dict = {
        'features': fts,
        'logit': logits
    }

    return saved_dict


def get_feature_logit(
    logger, eb_network, cur_n_cls, loader, device=None, save_dir=None, taski=0, loader_label="single",
):
    saved_dict = get_split_ft_logit_by_class(eb_network, cur_n_cls, loader, device)

    if save_dir is not None:
        ft_save_pth = os.path.join(save_dir, f"{loader_label}_step{taski}_ft.npy")
        np.save(ft_save_pth, saved_dict, allow_pickle=True)
        logger.info(f"Save feature in the {ft_save_pth}")


# -> generate logits directly from different part of extractors
def get_logits_distribution(net, cur_n_cls, loader, device=None):
    assert device is not None
    whole_logits = [[] for i in range(cur_n_cls)]
    old_logits = [[] for i in range(cur_n_cls)]
    new_logits = [[] for i in range(cur_n_cls)]
    first_group, first_lbl = None, None
    is_first = True

    net.eval()

    with torch.no_grad():
        for (x, y) in loader:
            if is_first:
                is_first = False
                first_group = x.cpu()
                first_lbl = y.cpu()

            x = x.to(device)
            opt_dict = net.split_forward(x)
            w_logit = opt_dict['logit']
            o_logit = opt_dict['old_logit']
            n_logit = opt_dict['new_logit']

            for lbl in range(cur_n_cls):
                old_logits[lbl].append(o_logit.cpu()[y == lbl])
                new_logits[lbl].append(n_logit.cpu()[y == lbl])
                whole_logits[lbl].append(w_logit.cpu()[y == lbl])

    for i in range(len(old_logits)):
        old_logits[i] = torch.cat((old_logits[i])).numpy()
        new_logits[i] = torch.cat((new_logits[i])).numpy()
        whole_logits[i] = torch.cat((whole_logits[i])).numpy()

    saved_dict = {
        'old_logit': old_logits,
        'new_logit': new_logits,
        'whole_logit': whole_logits,
        'first_gp': first_group,
        'first_lbl': first_lbl,
    }

    return saved_dict


def get_step_logits(
    logger, net, cur_n_cls, loader, device=None, save_dir=None, taski=0,
):
    assert save_dir is not None
    saved_dict = get_logits_distribution(net, cur_n_cls, loader, device=device)
    logit_save_pth = os.path.join(save_dir, f"step{taski}_logits_part.npy")
    np.save(logit_save_pth, saved_dict, allow_pickle=True)
    logger.info(f"Save two part logit distribution in the {logit_save_pth}")


# -> generate the spiking logits of different class
def get_spiking_logits_distribution(net, cur_n_cls, loader, device=None):
    assert device is not None
    whole_logits = [[] for i in range(cur_n_cls)]

    net.eval()

    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device)
            opt_dict = net(x)
            # -> tried to get a balanced logit
            w_logit = torch.mean(opt_dict['logit'], dim=1)

            for lbl in range(cur_n_cls):
                whole_logits[lbl].append(w_logit.cpu()[y == lbl])

    for i in range(len(whole_logits)):
        whole_logits[i] = torch.cat((whole_logits[i])).numpy()

    saved_dict = {
        'whole_logit': whole_logits,
    }

    return saved_dict


def get_spiking_step_logits(
    logger, net, cur_n_cls, loader, device=None, save_dir=None, taski=0,
):
    assert save_dir is not None
    saved_dict = get_spiking_logits_distribution(
        net, cur_n_cls, loader, device=device
    )
    logit_save_pth = os.path.join(save_dir, f"spiking_step{taski}_logits_part.npy")
    np.save(logit_save_pth, saved_dict, allow_pickle=True)
    logger.info(f"Save logit distribution in the {logit_save_pth}")


# -> generate the distribution among temporal dimension
def get_spiking_logits_temporal_distrib(net, cur_n_cls, loader, device: None):
    assert device is not None
    whole_logits = [[] for i in range(cur_n_cls)]
    net.eval()

    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device)
            opt_dict = net(x)
            t_logits = torch.mean(opt_dict['logit'], dim=2)

            for lbl in range(cur_n_cls):
                whole_logits[lbl].append(t_logits.cpu()[y == lbl])

    for i in range(len(whole_logits)):
        whole_logits[i] = torch.cat((whole_logits[i])).numpy()

    saved_dict = {
        't_logits': whole_logits
    }
    return saved_dict


def get_spiking_temporal_logits(
    logger, net, cur_n_cls, loader, device=None, save_dir=None, taski=0,
):
    assert save_dir is not None
    save_dict = get_spiking_logits_temporal_distrib(net, cur_n_cls, loader, device)
    logit_save_pth = os.path.join(save_dir, f"spiking_step{taski}_logits_temporal.npy")
    np.save(logit_save_pth, save_dict, allow_pickle=True)
    logger.info(f"Save logit distribution in the {logit_save_pth}")


##########################
# generate the ft and logit in different part
def get_different_distrib(
    net, old_net, train_loader, test_loader, last_n_cls, cur_n_cls, sample_num=100, device=None
):
    assert device is not None
    old_dim = net.last_dim
    new_dim = net.features_dim - old_dim

    label_name = ["test_e", "train_e", "test_t", "train_t"]
    dict_list = []

    for i in range(2):
        for j in range(2):
            ############################
            # different setting of the network and the loader
            if i == 0:
                net.eval()
            else:
                net.train()
            if j == 0:
                loader = test_loader
            else:
                loader = train_loader

            fts = [[] for i in range(cur_n_cls)]
            old_logits = [[] for i in range(cur_n_cls)]
            new_logits = [[] for i in range(cur_n_cls)]

            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    opt_dict = net.split_forward(x)
                    for lbl in range(cur_n_cls):
                        fts[lbl].append(opt_dict['feature'].cpu()[y == lbl])
                        old_logits[lbl].append(opt_dict['old_logit'].cpu()[y == lbl])
                        new_logits[lbl].append(opt_dict['new_logit'].cpu()[y == lbl])

            for i in range(len(fts)):
                fts[i] = torch.cat((fts[i])).numpy()
                fts[i] = fts[i][:min(sample_num, len(fts[i]))]

                old_logits[i] = torch.cat((old_logits[i])).numpy()
                old_logits[i] = old_logits[i][:min(sample_num, len(old_logits[i]))]

                new_logits[i] = torch.cat((new_logits[i])).numpy()
                new_logits[i] = new_logits[i][:min(sample_num, len(new_logits[i]))]

            one_mode_dict = {
                'feature': fts,
                'old_logit': old_logits,
                'new_logit': new_logits,
            }
            dict_list.append(one_mode_dict)

    opt_dict = {
        'cur_n_cls': cur_n_cls,
        'last_n_cls': last_n_cls,
        'old_dim': old_dim,
        'new_dim': new_dim,
        label_name[0]: None,
        label_name[1]: None,
        label_name[2]: None,
        label_name[3]: None,
    }
    assert len(label_name) == len(dict_list)
    for i in range(4):
        opt_dict[label_name[i]] = dict_list[i]

    return opt_dict


def get_single_distrib(
    net, train_loader, test_loader, last_n_cls, cur_n_cls, sample_num=100, device=None
):
    assert device is not None
    old_dim = 0
    new_dim = net.features_dim

    label_name = ["test_e", "train_e", "test_t", "train_t"]
    dict_list = []

    for i in range(2):
        for j in range(2):
            ############################
            # different setting of the network and the loader
            if i == 0:
                net.eval()
            else:
                net.train()
            if j == 0:
                loader = test_loader
            else:
                loader = train_loader

            fts = [[] for i in range(cur_n_cls)]
            logits = [[] for i in range(cur_n_cls)]

            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    opt_dict = net.forward(x)
                    for lbl in range(cur_n_cls):
                        fts[lbl].append(opt_dict['feature'].cpu()[y == lbl])
                        logits[lbl].append(opt_dict['logit'].cpu()[y == lbl])

            for i in range(len(fts)):
                fts[i] = torch.cat((fts[i])).numpy()
                fts[i] = fts[i][:min(sample_num, len(fts[i]))]

                logits[i] = torch.cat((logits[i])).numpy()
                logits[i] = logits[i][:min(sample_num, len(logits[i]))]

            one_mode_dict = {
                'feature': fts,
                'logit': logits,
            }
            dict_list.append(one_mode_dict)

    opt_dict = {
        'cur_n_cls': cur_n_cls,
        'last_n_cls': 0,
        'old_dim': old_dim,
        'new_dim': new_dim,
        label_name[0]: None,
        label_name[1]: None,
        label_name[2]: None,
        label_name[3]: None,
    }
    assert len(label_name) == len(dict_list)
    for i in range(4):
        opt_dict[label_name[i]] = dict_list[i]

    return opt_dict


def get_whole_features_distribution(
    logger, new_network, old_network, last_n_cls, cur_n_cls, test_loader, train_loader,
    sample_num, device=None, save_dir=None, taski=0,
):
    if taski == 0:
        opt_dict = get_single_distrib(
            new_network, train_loader, test_loader, last_n_cls, cur_n_cls,
            sample_num, device=device
        )
    else:
        opt_dict = get_different_distrib(
            new_network, old_network, train_loader, test_loader, last_n_cls, cur_n_cls,
            sample_num, device=device
        )

    if save_dir is not None:
        logit_save_pth = os.path.join(save_dir, f"step{taski}_train_eval_distrib.npy")
        np.save(logit_save_pth, opt_dict, allow_pickle=True)
        logger.info(f"Save the different distribution of the network in the {logit_save_pth}")


##########################
# The part concerning about plotting and analysing (of classifier weight)
##########################
def check_clf_weight_distrib(network, new_cls_size, new_ft_size, logger):
    clf_w = network.module.classifier.weight.data
    # -> weight shape <cls_dim, ft_dim>
    cls_dim, ft_dim = clf_w.shape[0], clf_w.shape[1]
    old_cls_size = cls_dim - new_cls_size
    old_ft_size = ft_dim - new_ft_size
    o2o_mean = torch.mean(torch.square(clf_w[:old_cls_size, :old_ft_size])).item()
    o2n_mean = torch.mean(torch.square(clf_w[old_cls_size:, :old_ft_size])).item()
    n2o_mean = torch.mean(torch.square(clf_w[:old_cls_size, old_ft_size:])).item()
    n2n_mean = torch.mean(torch.square(clf_w[old_cls_size:, old_ft_size:])).item()

    o2o_abs = torch.mean(torch.abs(clf_w[:old_cls_size, :old_ft_size])).item()
    o2n_abs = torch.mean(torch.abs(clf_w[old_cls_size:, :old_ft_size])).item()
    n2o_abs = torch.mean(torch.abs(clf_w[:old_cls_size, old_ft_size:])).item()
    n2n_abs = torch.mean(torch.abs(clf_w[old_cls_size:, old_ft_size:])).item()

    o2o_cmm = torch.mean(clf_w[:old_cls_size, :old_ft_size]).item()
    o2n_cmm = torch.mean(clf_w[old_cls_size:, :old_ft_size]).item()
    n2o_cmm = torch.mean(clf_w[:old_cls_size, old_ft_size:]).item()
    n2n_cmm = torch.mean(clf_w[old_cls_size:, old_ft_size:]).item()

    logger.info("Several value of four weight block:")
    logger.info(f"Square_mean: o2o: {o2o_mean:.3f}, o2n: {o2n_mean:.3f}, n2o: {n2o_mean:.3f}, n2n: {n2n_mean:.3f}")
    logger.info(f"Abs_mean: o2o: {o2o_abs:.3f}, o2n: {o2n_abs:.3f}, n2o: {n2o_abs:.3f}, n2n: {n2n_abs:.3f}")
    logger.info(f"Val_mean: o2o: {o2o_cmm:.3f}, o2n: {o2n_cmm:.3f}, n2o: {n2o_cmm:.3f}, n2n: {n2n_cmm:.3f}")


def check_clf_weight_distrib_specific(network, new_cls_size, new_ft_size, cls_dim, logger):
    clf_w = network.module.classifier.weight.data
    # -> weight shape <cls_dim, ft_dim>
    ft_dim = clf_w.shape[1]
    old_cls_size = cls_dim - new_cls_size
    old_ft_size = ft_dim - new_ft_size
    o2o_mean = torch.mean(torch.square(clf_w[:old_cls_size, :old_ft_size])).item()
    o2n_mean = torch.mean(torch.square(clf_w[old_cls_size:cls_dim, :old_ft_size])).item()
    n2o_mean = torch.mean(torch.square(clf_w[:old_cls_size, old_ft_size:])).item()
    n2n_mean = torch.mean(torch.square(clf_w[old_cls_size:cls_dim, old_ft_size:])).item()

    o2o_abs = torch.mean(torch.abs(clf_w[:old_cls_size, :old_ft_size])).item()
    o2n_abs = torch.mean(torch.abs(clf_w[old_cls_size:cls_dim, :old_ft_size])).item()
    n2o_abs = torch.mean(torch.abs(clf_w[:old_cls_size, old_ft_size:])).item()
    n2n_abs = torch.mean(torch.abs(clf_w[old_cls_size:cls_dim, old_ft_size:])).item()

    o2o_cmm = torch.mean(clf_w[:old_cls_size, :old_ft_size]).item()
    o2n_cmm = torch.mean(clf_w[old_cls_size:cls_dim, :old_ft_size]).item()
    n2o_cmm = torch.mean(clf_w[:old_cls_size, old_ft_size:]).item()
    n2n_cmm = torch.mean(clf_w[old_cls_size:cls_dim, old_ft_size:]).item()

    logger.info("Several value of four weight block:")
    logger.info(f"o2o: {o2o_mean:.3f}, o2n: {o2n_mean:.3f}, n2o: {n2o_mean:.3f}, n2n: {n2n_mean:.3f}")
    logger.info(f"Abs_mean: o2o: {o2o_abs:.3f}, o2n: {o2n_abs:.3f}, n2o: {n2o_abs:.3f}, n2n: {n2n_abs:.3f}")
    logger.info(f"Val_mean: o2o: {o2o_cmm:.3f}, o2n: {o2n_cmm:.3f}, n2o: {n2o_cmm:.3f}, n2n: {n2n_cmm:.3f}")
