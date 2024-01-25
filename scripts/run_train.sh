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
--lr=5e-4 \
--num_warmup_steps=10000 \
--device='gpu' \
--distributed \
--cylambda1 0.25 \
--cylambda2 0.25


--checkpoint='' \
--pretrained=''