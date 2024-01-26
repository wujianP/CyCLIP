import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

# from dataset_zoo import get_data
# from model_zoo.clip_models import CLIPWrapper


@torch.no_grad()
def vl_eval(args, model, preprocess, tokenizer, device="cuda"):
    pass
    # # wrap model
    # clip_model = CLIPWrapper(model, device, None)
    #
    # # load data
    # data = get_data(data_names=args.val_vl_data_type,
    #                 data_root=args.val_vl_data_root,
    #                 image_preprocess=preprocess,
    #                 tokenizer=tokenizer)
    #
    # result = {}
    # for bench in data:
    #     _, acc = clip_model.evaluate(benchmark=bench, batch_size=args.val_batch_size, num_workers=args.val_workers)
    #     result[bench['name']] = {
    #         'i2t_acc': acc["i2t_accuracy"],
    #         't2i_acc': acc["t2i_accuracy"],
    #     }
    #
    # return result