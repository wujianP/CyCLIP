cd /discobox/wjpeng/code/2024/CyCLIP/utils
conda activate /discobox/wjpeng/env/cyCLIP

python download.py \
--file='/discobox/wjpeng/dataset/cc3m/train_03.tsv' --dir='/discobox/wjpeng/dataset/cc3m/data/train'


#split -n l/4 -d /discobox/wjpeng/dataset/cc3m/train.tsv train_ --additional-suffix=.tsv
