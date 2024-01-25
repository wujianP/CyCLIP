cd /discobox/wjpeng/code/2024/CyCLIP
conda activate /discobox/wjpeng/env/cyCLIP
rm -rf /discobox/wjpeng/ckp/betterCLIP/debug
python -m src.main \
--log='/discobox/wjpeng/ckp/betterCLIP/debug' \
--name='exp1' \
--train_data='/discobox/wjpeng/dataset/cc3m/data/train/train_01.csv' \
--validation_data='/discobox/wjpeng/dataset/cc3m/data/val/train.csv' \
--image_key='image' \
--caption_key='caption' \
--device_ids 0 1 2 3 4 5 6 7 \
--model_name='RN50' \
--epochs='1' \
--batch_size=128 \
--lr=5e-6 \
--num_warmup_steps=10000 \
--device='gpu' \
--distributed \
--cylambda1 0.25 \
--cylambda2 0.25 \
--extra-train-data='' \
--extra-batch-size=8 \
--extra-data-type


--checkpoint='' \
--pretrained=''


# 单卡debug
cd /discobox/wjpeng/code/2024/CyCLIP
conda activate /discobox/wjpeng/env/cyCLIP
rm -rf /discobox/wjpeng/ckp/betterCLIP/debug
python -m src.main \
--log='/discobox/wjpeng/ckp/betterCLIP/debug' \
--name='exp1' \
--train_data='/discobox/wjpeng/dataset/cc3m/data/train/train_all_filtered.csv' \
--validation_data='/discobox/wjpeng/dataset/cc3m/data/val/train_filtered.csv' \
--from-pretrained='/DDN_ROOT/wjpeng/weights/cyclip/cyclip.pt' \
--image_key='image' \
--caption_key='caption' \
--device_ids 1 \
--model_name='RN50' \
--epochs='1' \
--batch_size=16 \
--lr=5e-4 \
--num_warmup_steps=10000 \
--device='gpu' \
--cylambda1 0.25 \
--cylambda2 0.25 \
--extra-train-data='/DDN_ROOT/wjpeng/dataset/VL-benchmark/train/' \
--extra-batch-size=8 \
--extra-data-type count existence relative-size absolute-size absolute-spatial relative-spatial

