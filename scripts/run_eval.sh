export CUDA_VISIBLE_DEVICES=0
cd /discobox/wjpeng/code/2024/CyCLIP
conda activate /discobox/wjpeng/env/cyCLIP
python -m src.main \
--train_data='/discobox/wjpeng/dataset/cc3m/data/val/train_filtered.csv' \
--extra-train-data='/DDN_ROOT/wjpeng/dataset/VL-benchmark/train/' \
--extra-data-type count existence relative-size absolute-size absolute-spatial relative-spatial \
--extra-ann-root='/DDN_ROOT/wjpeng/dataset/VL-benchmark/train/captions_expanded/' \
--val-batch-size=8 \
--val-workers=8 \
--val-common-data-root='/DDN_ROOT/wjpeng/dataset/clip_benchmark_datasets' \
--val-vl-data-root='/DDN_ROOT/wjpeng/dataset/VL-benchmark/clean_val_v3/data' \
--logs='/discobox/wjpeng/ckp/betterCLIP/rebuttal/eval' \
--name='rn50-cyclip_ep10-step400_lr2e-6-warm1000_common-cc3m-bs64_extra-wt0.2-bs4-hn2-v2' \
--from-pretrained='/discobox/wjpeng/ckp/betterCLIP/rebuttal/rn50-cyclip_ep10-step400_lr2e-6-warm1000_common-cc3m-bs64_extra-wt0.2-bs4-hn2-v2/checkpoints/epoch_10.pt' \
--wandb \
--wandb-project-name='cyCLIP_EVAL' \
--wandb-key='8cff0498531e0409db5f3c43b52a26b0d068f2dc'
