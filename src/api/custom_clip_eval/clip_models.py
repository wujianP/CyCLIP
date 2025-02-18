import torch
from einops import rearrange

from tqdm import tqdm
from torch.utils.data import DataLoader


class CLIPWrapper:
    def __init__(self, model, device, tokenizer=None):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

    @staticmethod
    def cls_collate_fn(batch):
        images = []
        object_names = []
        labels = []
        for sp in batch:
            images.append(sp['image'])
            object_names.append(sp['object_names'])
            labels.append(sp['labels'])
        images = torch.stack(images, dim=0)
        return {
            'images': images,
            'object_names': object_names,
            'labels': labels
        }

    @torch.no_grad()
    def i2t_evaluate(self, benchmark, batch_size, num_workers):
        i2t_loader = DataLoader(dataset=benchmark['i2t_dataset'],
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                drop_last=False)
        tqdm_i2t_loader = tqdm(i2t_loader)
        tqdm_i2t_loader.set_description(f"Computing Image to Text retrieval scores ({benchmark['name']})")
        i2t_scores = []
        i2t_correct_num = 0
        for batch in tqdm_i2t_loader:
            bs = len(batch['label'])
            pixel_values = batch['pixel_values'].cuda()  # B,C,H,W (B:batch size)
            input_ids = batch['input_ids'].cuda()  # B,L,S (L:num of candidate texts, S:sentence length)
            attention_mask = batch['attention_mask'].cuda()  # B,L

            image_embeddings = self.model.get_image_features(pixel_values=pixel_values)  # B,D (D:feature dim)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

            input_ids = rearrange(input_ids, 'B L S -> (B L) S')
            attention_mask = rearrange(attention_mask, 'B L -> (B L)')
            text_embeddings = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)  # BL,D (D:feature dim)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = rearrange(text_embeddings, '(B L) D -> B L D', B=bs)

            batch_i2t_scores = torch.einsum('BD,BLD->BL', [image_embeddings, text_embeddings]).cpu()
            i2t_scores.append(batch_i2t_scores)

            gt_labels = torch.tensor(batch['label'])
            pred_labels = batch_i2t_scores.argmax(dim=-1)
            correct_num = (gt_labels == pred_labels).sum()
            i2t_correct_num += correct_num.item()

        i2t_scores = torch.cat(i2t_scores, dim=0)
        i2t_acc = 100 * i2t_correct_num / len(benchmark['i2t_dataset'])

        return i2t_scores, i2t_acc

    @torch.no_grad()
    def t2i_evaluate(self, benchmark, batch_size, num_workers):
        t2i_loader = DataLoader(dataset=benchmark['t2i_dataset'],
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                drop_last=False)
        tqdm_t2i_loader = tqdm(t2i_loader)
        tqdm_t2i_loader.set_description(f"Computing Text to Image retrieval scores ({benchmark['name']})")
        t2i_scores = []
        t2i_correct_num = 0
        for batch in tqdm_t2i_loader:
            bs = len(batch['label'])
            pixel_values = batch['pixel_values'].cuda()  # B,K,C,H,W (K:num of candidate images per case, S:sentence length)
            input_ids = batch['input_ids'].cuda()  # B,1,S (B:batch size, S:sentence length)
            attention_mask = batch['attention_mask'].cuda()  # B,1

            input_ids = rearrange(input_ids, 'B 1 S -> (B 1) S')
            attention_mask = rearrange(attention_mask, 'B 1 -> (B 1)')
            text_embeddings = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)  # B,D (D:feature dim)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

            pixel_values = rearrange(pixel_values, 'B K C H W -> (B K) C H W')
            image_embeddings = self.model.get_image_features(pixel_values=pixel_values)  # BK,D (K:num of candidate images per case, D:feature dim)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            image_embeddings = rearrange(image_embeddings, '(B K) D -> B K D', B=bs)

            batch_t2i_scores = torch.einsum('BD,BKD->BK', [text_embeddings, image_embeddings]).cpu()
            t2i_scores.append(batch_t2i_scores)

            gt_labels = torch.tensor(batch['label'])
            pred_labels = batch_t2i_scores.argmax(dim=-1)
            correct_num = (gt_labels == pred_labels).sum()
            t2i_correct_num += correct_num.item()

        t2i_scores = torch.cat(t2i_scores, dim=0)
        t2i_acc = 100 * t2i_correct_num / len(benchmark['t2i_dataset'])

        return t2i_scores, t2i_acc

    @torch.no_grad()
    def evaluate(self, benchmark, batch_size, num_workers):
        """Computes the match scores and accuracy for each image_option / caption_option pair.
        Args:
            benchmark (Dict): include "i2t_dataset" and "i2t_dataset"
            batch_size
            num_workers
        Returns:
            scores(Dict of Tensor): `i2t_scores` / `t2i_scores`
            accuracy(Dict of Scalar): `i2t_accuracy / t2i_accuracy`
        """

        # image to text retrieval
        i2t_scores, i2t_acc = self.i2t_evaluate(benchmark, batch_size, num_workers)

        # text to image retrieval
        t2i_scores, t2i_acc = self.t2i_evaluate(benchmark, batch_size, num_workers)

        """
        `i2t_scores`: tensor of shape NxL, N is the number of test samples, L is the number of candidate texts per sample
        `t2i_scores`: tensor of shape NxK, N is the number of test samples, K is the number of candidate images per sample
        """
        scores = {
            'i2t_scores': i2t_scores,
            't2i_scores': t2i_scores
        }

        accuracy = {
            'i2t_accuracy': i2t_acc,
            't2i_accuracy': t2i_acc,
        }

        return scores, accuracy
