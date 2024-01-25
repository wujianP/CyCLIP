import json
import random
import os
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image


class ExtraDataset(Dataset):
    def __init__(self, data_root, ann_root, hard_num, image_names, text_templates, object_num,
                 blip_caption=True, llama_caption=False, processor=None):
        """
        Args:
            data_root: the path to the root dir of dataset
            hard_num: number of hard negative pairs for each sample (used in training_ins)
            image_names: the list of image file names of each sample
            text_templates: text templates for the dataset
            object_num: one object or two object in the relation
        """
        self.data_root = data_root
        self.hard_num = hard_num
        self.image_names = image_names
        self.text_templates = text_templates
        self.object_num = object_num
        assert object_num in [1, 2], f"only support 1, 2 object, but got {object_num}."
        self.processor = processor
        self.sample_list = [sample for sample in os.listdir(self.data_root)
                            if os.path.isdir(os.path.join(self.data_root, sample))]
        with open(ann_root, 'r') as ann_file:
            self.ann_dict = json.load(ann_file)
        ann_file.close()

        self.blip_caption = blip_caption
        self.llama_caption = llama_caption

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        sample_dir = os.path.join(self.data_root, self.sample_list[idx])

        with open(os.path.join(sample_dir, 'metadata.json')) as f:
            metadata = json.load(f)
            if self.object_num == 1:
                try:
                    cls_nm = metadata['object']
                except:
                    cls_nm = metadata['removed-object']
            else:
                sub_nm = metadata['subject']
                obj_nm = metadata['object']
        f.close()

        image_list = []
        text_list = []
        for img_fn in random.sample(self.image_names, self.hard_num):

            # prepare image
            img_path = os.path.join(sample_dir, img_fn)
            img = Image.open(img_path).convert('RGB')
            image_list.append(img)

            # prepare text
            caption_info = self.ann_dict[os.path.join(self.sample_list[idx], img_fn)]
            if self.llama_caption:
                text = random.choice(caption_info['llama_caption'])
            else:
                if self.object_num == 1:
                    text = random.choice(self.text_templates[img_fn]).format(obj=cls_nm)
                else:
                    text = random.choice(self.text_templates[img_fn]).format(sub=sub_nm, obj=obj_nm)
            if self.blip_caption:
                blip_caption = caption_info['blip_captions']
                text = f'{blip_caption}, and {text}'
            text_list.append(text)

        images = torch.stack([self.processor.process_image(img) for img in image_list], dim=0)
        texts = self.processor.process_text(text_list)

        sample = {
            'pixel_values': images,
            'input_ids': texts['input_ids'],
            'attention_mask': texts['attention_mask']
        }

        return sample


def get_extra_data(args, data_type, processor):
    """get the dataloader for each dataset type"""

    if data_type == 'absolute-size':
        from .extra_dataset_metadata import absolute_size_metadata as metadata
    elif data_type == 'relative-size':
        from .extra_dataset_metadata import relative_size_metadata as metadata
    elif data_type == 'absolute-spatial':
        from .extra_dataset_metadata import absolute_spatial_metadata as metadata
    elif data_type == 'relative-spatial':
        from .extra_dataset_metadata import relative_spatial_metadata as metadata
    elif data_type == 'count':
        from .extra_dataset_metadata import count_metadata as metadata
    elif data_type == 'existence':
        from .extra_dataset_metadata import existence_metadata as metadata
    else:
        raise KeyError(f'Unknown dataset type {data_type}.')

    data_root = os.path.join(args.extra_train_data, data_type.replace('-', '_'))
    hard_num = 2
    assert hard_num <= metadata['set_size']

    dataset = ExtraDataset(
        data_root=data_root,
        ann_root=os.path.join(args.extra_ann_root, f"{data_type.replace('-', '_')}_captions_all.json"),
        hard_num=hard_num,
        image_names=metadata['image_names'],
        text_templates=metadata['text_templates'],
        object_num=metadata['object_num'],
        processor=processor
    )

    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.extra_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.extra_workers,
        drop_last=True,
        pin_memory=True)

    dataloader.num_samples = len(dataloader) * args.extra_batch_size
    dataloader.num_batches = len(dataloader)

    return dataloader
