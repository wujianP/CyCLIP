
cd /discobox/wjpeng/code/2024/CyCLIP
conda activate /discobox/wjpeng/env/cyCLIP
python -m src.main \
--logs='/discobox/wjpeng/ckp/betterCLIP/rebuttal' \
--name='rn50-cyclip_ep10-step200_lr1e-6-warm1000_common-cc3m-bs128_extra-wt0.2-bs8-hn2' \
--log-per-steps=20 \
--train_data='/discobox/wjpeng/dataset/cc3m/data/train/train_all_filtered.csv' \
--from-pretrained='/DDN_ROOT/wjpeng/weights/cyclip/cyclip.pt' \
--distributed \
--device_ids 0 1 2 3 4 5 6 7 \
--model_name='RN50' \
--batch_size=128 \
--lr=1e-6 \
--epochs=10 \
--steps-per-epoch=200 \
--num_warmup_steps=1000 \
--cylambda1 0.25 \
--cylambda2 0.25 \
--extra-train-data='/DDN_ROOT/wjpeng/dataset/VL-benchmark/train/' \
--extra-batch-size=8 \
--extra-workers=2 \
--extra-loss-wt=0.2 \
--extra-data-type count existence relative-size absolute-size absolute-spatial relative-spatial \
--extra-ann-root='/DDN_ROOT/wjpeng/dataset/VL-benchmark/train/captions_expanded/' \
--val-batch-size=32 \
--val-workers=6 \
--val-common-data-root='/DDN_ROOT/wjpeng/dataset/clip_benchmark_datasets' \
--val-vl-data-root='/DDN_ROOT/wjpeng/dataset/VL-benchmark/clean_val_v3/data' \
--wandb \
--wandb-project-name='test' \
--wandb-key='8cff0498531e0409db5f3c43b52a26b0d068f2dc'
 \
--do-not-eval-epoch-0








