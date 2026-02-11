import os
import numpy as np
from scipy.special import softmax
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from plotting_utils import plot_histogram, plot_histograms, plot_box

exp_tag = "S1-R0-bal_binary-bal_binary_aux_n1_cifar100_inc10b20_logit_ctrl"
taski = 1
initial_inc = 10
task_size = 10
n_class = taski * task_size + initial_inc
out_dim = 512  # -> 卷积的输出规格，换提取器的时候记得随时修改
plt_tag = "inc_data"
logit_name = f"{plt_tag}_step{taski}_ft.npy"
binary_logit_name = f"step{taski}_binary.npy"

cur_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(cur_dir)
ckpt_pth = os.path.join(project_dir, f"store/opt_models/{exp_tag}/ckpts")
binary_file_pth = os.path.join(ckpt_pth, binary_logit_name)
file_pth = os.path.join(ckpt_pth, logit_name)

plotting_pth = os.path.join(project_dir, f"store/figures")
plotting_pth = os.path.join(plotting_pth, exp_tag)

if __name__ == "__main__":
    info_dict = np.load(file_pth, allow_pickle=True)
    info_dict = info_dict.item()

    fts = info_dict['features']
    logits = info_dict['logit']

    if not os.path.exists(plotting_pth):
        os.mkdir(plotting_pth)

    old_summ = n_class - task_size
    ###########################################
    # single logit part
    old_gt_logit, new_gt_logit = [], []
    for i in range(old_summ):
        logits[i] = softmax(logits[i], axis=1)
        # old_gt_logit.append(logits[i][:, i])
        old_gt_logit.append(np.sum(logits[i][:, old_summ:], axis=1))
        # old_gt_logit.append(np.sum(logits[i][:, :old_summ], axis=1) - logits[i][:, i])
    old_gt_logit = np.concatenate(old_gt_logit)
    for i in range(old_summ, n_class):
        logits[i] = softmax(logits[i], axis=1)
        # new_gt_logit.append(logits[i][:, i])
        new_gt_logit.append(np.sum(logits[i][:, :old_summ], axis=1))
        # new_gt_logit.append(np.sum(logits[i][:, old_summ:], axis=1) - logits[i][:, i])
    new_gt_logit = np.concatenate(new_gt_logit)
    print(f"old_mean: {np.mean(old_gt_logit):.4f}, new_mean: {np.mean(new_gt_logit):.4f}")
    print(f"old_std: {np.std(old_gt_logit):.3f}, new_std: {np.std(new_gt_logit):.3f}")
    print(f"old_shape: {old_gt_logit.shape}, new_shape: {new_gt_logit.shape}")

    old_grp = []
    sample_size = new_gt_logit.shape[0]
    for i in range(taski - 1):
        if i == 0:
            old_grp.append(old_gt_logit[-(i+1) * sample_size:])
        else:
            old_grp.append(old_gt_logit[-(i+1) * sample_size:-i * sample_size])
        print(old_grp[-1].shape)
    if initial_inc >= task_size:
        if taski == 1:
            old_grp.append(old_gt_logit[:])
        else:
            old_grp.append(old_gt_logit[:-(taski - 1) * sample_size])
    print(old_grp[-1].shape)
    whole_logit = np.stack([*old_grp, new_gt_logit])
    print(whole_logit.shape)
    fig_old = plot_histograms(whole_logit, bins=50)
    fig_old_name = f"{exp_tag}_{plt_tag}_step-{taski}"
    old_save_pth = os.path.join(plotting_pth, fig_old_name)
    fig_old.savefig(old_save_pth)

    ###########################################
    # single feature part
    old_ft_norm, new_ft_norm = [], []
    old_ft_past, new_ft_past = [], []
    old_ft_curr, new_ft_curr = [], []
    for i in range(old_summ):
        old_ft_norm.append(np.linalg.norm(fts[i], axis=1))
        old_ft_past.append(np.linalg.norm(fts[i][:, :-out_dim], axis=1))
        old_ft_curr.append(np.linalg.norm(fts[i][:, -out_dim:], axis=1))
    old_ft_norm = np.concatenate(old_ft_norm)
    old_ft_past = np.concatenate(old_ft_past)
    old_ft_curr = np.concatenate(old_ft_curr)
    for i in range(old_summ, n_class):
        new_ft_norm.append(np.linalg.norm(fts[i], axis=1))
        new_ft_past.append(np.linalg.norm(fts[i][:, :-out_dim], axis=1))
        new_ft_curr.append(np.linalg.norm(fts[i][:, -out_dim:], axis=1))
    new_ft_norm = np.concatenate(new_ft_norm)
    new_ft_past = np.concatenate(new_ft_past)
    new_ft_curr = np.concatenate(new_ft_curr)
    print(f"whole old shape: {old_ft_norm.shape}, whole new shape: {new_ft_norm.shape}")
    print(f"whole old norm: {np.mean(old_ft_norm):.3f}, whole new norm: {np.mean(new_ft_norm):.3f}")
    print(f"past old norm: {np.mean(old_ft_past):.3f}, past new norm: {np.mean(new_ft_past):.3f}")
    print(f"cur old norm: {np.mean(old_ft_curr):.3f}, cur new norm: {np.mean(new_ft_curr):.3f}")

    old_norm_grp = []
    old_past_norm_grp = []
    old_curr_norm_grp = []
    sample_size = new_ft_norm.shape[0]
    for i in range(taski - 1):
        if i == 0:
            old_norm_grp.append(old_ft_norm[-(i+1) * sample_size:])
            old_past_norm_grp.append(old_ft_past[-(i+1) * sample_size:])
            old_curr_norm_grp.append(old_ft_curr[-(i+1) * sample_size:])
        else:
            old_norm_grp.append(old_ft_norm[-(i+1) * sample_size:-i * sample_size])
            old_past_norm_grp.append(old_ft_past[-(i + 1) * sample_size:-i * sample_size])
            old_curr_norm_grp.append(old_ft_curr[-(i + 1) * sample_size:-i * sample_size])
        print(old_norm_grp[-1].shape)
    if initial_inc >= task_size:
        if taski == 1:
            old_norm_grp.append(old_ft_norm[:])
            old_past_norm_grp.append(old_ft_past[:])
            old_curr_norm_grp.append(old_ft_curr[:])
        else:
            old_norm_grp.append(old_ft_norm[:-(taski - 1) * sample_size])
            old_past_norm_grp.append(old_ft_past[:-(taski - 1) * sample_size])
            old_curr_norm_grp.append(old_ft_curr[:-(taski - 1) * sample_size])
    print(old_norm_grp[-1].shape)
    whole_norm = np.stack([*old_norm_grp, new_ft_norm])
    past_norm = np.stack([*old_past_norm_grp, new_ft_past])
    curr_norm = np.stack([*old_curr_norm_grp, new_ft_curr])
    print(whole_norm.shape)
    fig_whole_norm = plot_histograms(whole_norm, bins=50)
    fig_whole_name = f"{exp_tag}_{plt_tag}_whole-norm_step-{taski}"
    fig_past_norm = plot_histograms(past_norm, bins=50)
    fig_past_name = f"{exp_tag}_{plt_tag}_past-norm_step-{taski}"
    fig_curr_norm = plot_histograms(curr_norm, bins=50)
    fig_curr_name = f"{exp_tag}_{plt_tag}_curr-norm_step-{taski}"

    fig_whole_norm.savefig(os.path.join(plotting_pth, fig_whole_name))
    fig_past_norm.savefig(os.path.join(plotting_pth, fig_past_name))
    fig_curr_norm.savefig(os.path.join(plotting_pth, fig_curr_name))

    # ###########################################
    # # single logit part of last part
    # last_logit_name = f"{plt_tag}_step{taski-1}_ft.npy"
    # last_file_pth = os.path.join(ckpt_pth, last_logit_name)
    #
    # last_info_dict = np.load(last_file_pth, allow_pickle=True)
    # last_info_dict = last_info_dict.item()
    #
    # last_logits = last_info_dict['logit']
    # src_logit = []
    # for i in range(old_summ):
    #     src_logit.append(last_logits[i][:, i])
    # src_logit = np.concatenate(src_logit)
    # print(f"src_mean: {np.mean(src_logit):.3f}, src_std: {np.std(src_logit):.3f}")
    # new_old_logit = np.stack((old_gt_logit, new_gt_logit, src_logit))
    # fig_mixed = plot_histograms(new_old_logit, bins=50)
    # fig_mixed_name = f"{exp_tag}_{plt_tag}_mixed-{taski-1}-{taski}"
    # mixed_save_pth = os.path.join(plotting_pth, fig_mixed_name)
    # fig_mixed.savefig(mixed_save_pth)

    # #############################################
    # # Just for consideration
    # old_summ = n_class - task_size
    # from scipy.special import softmax
    # new_credit = []
    # target_static = np.zeros(old_summ,)
    # for i in range(old_summ, n_class):
    #     credit = softmax(logits[i][:, :old_summ], axis=-1)
    #     max_target = np.max(credit, axis=-1)
    #     max_arg = np.argmax(credit, axis=-1)
    #     new_credit.append(np.mean(max_target))
    #     for i in range(max_arg.shape[0]):
    #         target_static[max_arg[i]] += 1
    # print(new_credit)
    # print(target_static)

    # ###########################################
    # # single feature part
    # ft_sum = []
    # for i in range(len(fts)):
    #     aver_distrib = np.mean(fts[i], axis=0)
    #     fig = plot_histogram(aver_distrib)
    #
    #     save_name = f"{exp_tag}_step-{taski}_class-{i}.png"
    #     save_pth = os.path.join(plotting_pth, save_name)
    #     fig.savefig(save_pth)
    #
    #     ft_sum.append(aver_distrib)
    #
    # ft_sum = np.stack(ft_sum)
    # ft_sum = np.mean(ft_sum, axis=0)
    # print(ft_sum.shape)
    # fig = plot_histogram(ft_sum)
    # save_name = f"{exp_tag}_step-{taski}"
    # save_pth = os.path.join(plotting_pth, save_name)
    # fig.savefig(save_pth)

    # ###########################################
    # # two part feature part
    # ft_old_sum = []
    # ft_new_sum = []
    # for i in range(len(fts)):
    #     aver_old_distrib = np.mean(fts[i][:, :-out_dim], axis=0)
    #     aver_new_distrib = np.mean(fts[i][:, -out_dim:], axis=0)
    #
    #     fig1 = plot_histogram(aver_old_distrib)
    #     fig2 = plot_histogram(aver_new_distrib)
    #
    #     ft_old_sum.append(aver_old_distrib)
    #     ft_new_sum.append(aver_new_distrib)
    #
    #     save_name_1 = f"{exp_tag}_step-{taski}_class-{i}_old.png"
    #     save_pth_1 = os.path.join(plotting_pth, save_name_1)
    #     fig1.savefig(save_pth_1)
    #
    #     save_name_2 = f"{exp_tag}_step-{taski}_class-{i}_new.png"
    #     save_pth_2 = os.path.join(plotting_pth, save_name_2)
    #     fig2.savefig(save_pth_2)
    #
    #     plt.close()
    #
    # ft_old_sum = np.stack(ft_old_sum)
    # ft_old_sum = np.mean(ft_old_sum, axis=0)
    # print(ft_old_sum.shape)
    # fig1 = plot_histogram(ft_old_sum)
    # save_name_1 = f"{exp_tag}_step-{taski}_old.png"
    # save_pth_1 = os.path.join(plotting_pth, save_name_1)
    # fig1.savefig(save_pth_1)
    #
    # ft_new_sum = np.stack(ft_new_sum)
    # ft_new_sum = np.mean(ft_new_sum, axis=0)
    # print(ft_new_sum.shape)
    # fig2 = plot_histogram(ft_new_sum)
    # save_name_2 = f"{exp_tag}_step-{taski}_new.png"
    # save_pth_2 = os.path.join(plotting_pth, save_name_2)
    # fig2.savefig(save_pth_2)

    # ###########################################
    # # binary part
    # binary_info_dict = np.load(binary_file_pth, allow_pickle=True)
    # binary_info_dict = binary_info_dict.item()
    #
    # def sigmoid(x):
    #     return 1 / (1 + np.exp(-x))
    #
    # binary_logits = binary_info_dict['logit']
    # for i in range(len(binary_logits)):
    #     binary_logits[i] = sigmoid(binary_logits[i])
    # fig = plot_box(data=binary_logits)
    # save_name = f"{exp_tag}_step-{taski}_binary2.png"
    # save_pth = os.path.join(plotting_pth, save_name)
    # fig.savefig(save_pth)
    #


