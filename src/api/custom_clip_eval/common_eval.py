"""using the toolkit from: https://github.com/LAION-AI/CLIP_benchmark/tree/main"""
import os
import torch
import json


@torch.no_grad()
def common_eval(model_name,
                pretrained,
                data_root,
                quiet=True,
                distributed=False,
                out_root=None,
                batch_size=64,):

    dataset = ['caltech101', 'flickr30k', 'mscoco_captions', 'cifar10', 'cifar100', 'imagenet1k', 'pets', 'stl10', 'flowers']

    if out_root is None:
        home_dir = os.path.expanduser("~")
        out_root = os.path.join(home_dir, '.cache', 'cyclip')

    os.makedirs(out_root, exist_ok=True)

    dataset = " ".join(dataset)

    # evaluate using CLI
    cmd = f'clip_benchmark eval \
    --model {model_name} \
    --pretrained {pretrained} \
    --recall_k 1 \
    --dataset {dataset} \
    --batch_size {batch_size} \
    --dataset_root "{data_root}/{{dataset}}" \
    --output "{out_root}/{{dataset}}.json"'

    if quiet:
        cmd = cmd + ' --quiet'
    if distributed:
        cmd = cmd + ' --distributed'

    os.system(cmd)

    # retrieve the evaluation result
    eval_result = {}
    for ret_fn in os.listdir(out_root):
        if ret_fn.endswith('.json'):
            with open(os.path.join(out_root, ret_fn), 'r') as f:
                result_info = json.load(f)
            f.close()

            dataset_name = result_info['dataset']
            eval_result[dataset_name] = [f'{k}: {v*100:.2f}' for k, v in result_info['metrics'].items()
                                         if k in ['acc1', 'image_retrieval_recall@1', 'text_retrieval_recall@1']]

    return eval_result
