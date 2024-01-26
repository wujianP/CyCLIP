# 单卡debug
cd /discobox/wjpeng/code/2024/CyCLIP
conda activate /discobox/wjpeng/env/cyCLIP
#rm -rf /discobox/wjpeng/ckp/betterCLIP/debug
python -m src.main \
--logs='/discobox/wjpeng/ckp/betterCLIP/debug' \
--name='exp5' \
--train_data='/discobox/wjpeng/dataset/cc3m/data/val/train_filtered.csv' \
--validation_data='/discobox/wjpeng/dataset/cc3m/data/val/train_filtered.csv' \
--from-pretrained='/DDN_ROOT/wjpeng/weights/cyclip/cyclip.pt' \
--image_key='image' \
--caption_key='caption' \
--distributed \
--device_ids 0 1 2 3 4 5 6 7 \
--model_name='RN50' \
--batch_size=64 \
--lr=5e-6 \
--num_warmup_steps=500 \
--device='gpu' \
--cylambda1 0.25 \
--cylambda2 0.25 \
--log-per-steps=10 \
--epochs=10 \
--steps-per-epoch=100 \
--extra-train-data='/DDN_ROOT/wjpeng/dataset/VL-benchmark/train/' \
--extra-batch-size=8 \
--extra-loss-wt=0.2 \
--extra-data-type count existence relative-size absolute-size absolute-spatial relative-spatial \
--extra-ann-root='/DDN_ROOT/wjpeng/dataset/VL-benchmark/train/captions_expanded/' \
--save_per_epoch=1 \
--save_most_recent \
--val-batch-size=64 \
--val-workers=8 \
--val-common-data-root='/DDN_ROOT/wjpeng/dataset/clip_benchmark_datasets' \
--val-vl-data-root='/DDN_ROOT/wjpeng/dataset/VL-benchmark/clean_val_v3/data' \
--wandb \
--wandb-project-name='test'






