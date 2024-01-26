import os
import json
import torch

from torch.utils.data import Dataset
from PIL import Image


class BaseDataset(Dataset):
    def __init__(self, data_root, ann, processor, mini_dataset=False):
        """
        Args:
            data_root: the path to the root dir of data
            ann: the path to the annotation file
        """
        self.data_root = data_root
        self.processor = processor
        self.mini_dataset = mini_dataset
        with open(ann, 'r') as f:
            self.sample_list = json.load(f)
        f.close()
        if self.mini_dataset:
            self.sample_list = self.sample_list[:20]

    def __len__(self):
        return len(self.sample_list)


class Image2TextDataset(BaseDataset):
    def __init__(self, data_root, processor, mini_dataset):
        ann = os.path.join(data_root, 'image2text_anns.json')
        super().__init__(data_root, ann, processor, mini_dataset)

    def __getitem__(self, idx):
        sample_info = self.sample_list[idx]

        # query image
        image_path = os.path.join(self.data_root, sample_info['query'])
        query_image = Image.open(image_path).convert('RGB')
        pixel_values = self.processor.process_image(query_image)

        # candidate texts
        candidate_texts = sample_info['keys']
        candidate_texts = self.processor.process_text(candidate_texts)
        input_ids = candidate_texts["input_ids"]
        attention_mask = candidate_texts["attention_mask"]

        # label
        label = sample_info['label']

        sample = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }

        return sample


class Text2ImageDataset(BaseDataset):
    def __init__(self, data_root, processor, mini_dataset):
        ann = os.path.join(data_root, 'text2image_anns.json')
        super().__init__(data_root, ann, processor, mini_dataset)

    def __getitem__(self, idx):
        sample_info = self.sample_list[idx]

        # query text
        query_text = sample_info['query']
        query_text = self.processor.process_text(query_text)
        input_ids = query_text["input_ids"]
        attention_mask = query_text["attention_mask"]

        # candidate images
        pixel_values = []
        for img in sample_info['keys']:
            img = Image.open(os.path.join(self.data_root, img)).convert('RGB')
            img = self.processor.process_image(img)
            pixel_values.append(img)
        pixel_values = torch.stack(pixel_values, dim=0)

        # label
        label = sample_info['label']

        sample = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }

        return sample


def get_data(data_root, processor, mini_dataset):
    """return the datasets for selected evaluation-real benchmarks"""

    data = []
    for dn in ['count', 'existence', 'absolute-spatial', 'relative-spatial', 'absolute-size', 'relative-size']:

        ann_filename = dn.replace('-', '_')
        i2t_dataset = Image2TextDataset(data_root=os.path.join(data_root, ann_filename), processor=processor, mini_dataset=mini_dataset)
        t2i_dataset = Text2ImageDataset(data_root=os.path.join(data_root, ann_filename), processor=processor, mini_dataset=mini_dataset)
        bench = {
            'name': dn,
            'i2t_dataset': i2t_dataset,
            't2i_dataset': t2i_dataset
        }

        data.append(bench)

    return data
