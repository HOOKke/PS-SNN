# https_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128 /jizhicfs/berlinni/miniconda3/envs/alade/bin/python main_exp.py --options options/data/cifar100_1order.yaml options/spiking_eb/spiking_allcomponents_cifar100_B0.yaml \
#     --initial-increment 0 --increment 20 \
#     --device 0 --label b0_5steps_orth_init_fix --no-benchmark \
#     --orthogonal_init_clf --fix_clf


# https_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128 python main_exp.py --options options/data/cifar100_3orders.yaml options/spiking_eb/spiking_allcomponents_cifar100_B0.yaml \
#     --initial-increment 0 --increment 10 \
#     --device 0 --label alade_snn_cifar100_b0_5steps --no-benchmark --seed-range 0 0


# CUDA_VISIBLE_DEVICES=0 https_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128 /jizhicfs/berlinni/miniconda3/envs/alade/bin/python main_exp.py --options options/data/cifar100_1order.yaml options/spiking_eb/spiking_allcomponents_cifar100_B0.yaml \
#     --initial-increment 0 --increment 20 \
#     --device 0 --label b0_5steps_orth_init --no-benchmark \
#     --orthogonal_init_clf


# CUDA_VISIBLE_DEVICES=1 https_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128 /jizhicfs/berlinni/miniconda3/envs/alade/bin/python main_exp.py --options options/data/cifar100_1order.yaml options/spiking_eb/spiking_allcomponents_cifar100_B0.yaml \
#     --initial-increment 0 --increment 20 \
#     --device 0 --label b0_5steps_orth_init_fix_200ep --no-benchmark \
#     --orthogonal_init_clf --fix_clf

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 https_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128 /jizhicfs/berlinni/miniconda3/envs/alade/bin/python main_exp.py --options options/data/cifar100_1order.yaml options/spiking_eb/spiking_allcomponents_cifar100_B0.yaml \
#     --initial-increment 0 --increment 5 \
#     --device 0 1 2 3 4 5 6 7 --label b0_20steps_orth_init --no-benchmark \
#     --orthogonal_init_clf

# CUDA_VISIBLE_DEVICES=0,1,2,3 https_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128 /jizhicfs/berlinni/miniconda3/envs/alade/bin/python main_exp.py --options options/data/cifar100_1order.yaml options/spiking_eb/spiking_allcomponents_cifar100_B0.yaml \
#     --initial-increment 0 --increment 10 \
#     --device 0 1 2 3 --label b0_10steps_orth_init --no-benchmark \
#     --orthogonal_init_clf

CUDA_VISIBLE_DEVICES=0 https_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128 /jizhicfs/berlinni/miniconda3/envs/alade/bin/python main_exp.py --options options/data/cifar100_1order.yaml options/spiking_eb/spiking_allcomponents_cifar100_B0.yaml \
    --initial-increment 0 --increment 20 \
    --device 0 --label b0_5steps_orth_init_clsfine_ --no-benchmark \
    --orthogonal_init_clf
