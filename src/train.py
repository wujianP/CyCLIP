import time
import wandb
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
from einops import rearrange


def get_loss(umodel, image_embeds, text_embeds, criterion, options):
    if options.distributed:
        if options.inmodal:
            raise KeyError
        else:
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]

            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)

            image_embeds = torch.cat(
                gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds = torch.cat(
                gathered_text_embeds[:options.rank] + [text_embeds] + gathered_text_embeds[options.rank + 1:])

    logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_image_per_text = logits_text_per_image.t()

    if options.inmodal:
        raise KeyError

    batch_size = len(logits_text_per_image)

    target = torch.arange(batch_size).long().to(options.device, non_blocking=True)

    contrastive_loss = torch.tensor(0).to(options.device)
    if options.inmodal:
        raise KeyError
    else:
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text,
                                                                                            target)) / 2
        contrastive_loss = crossmodal_contrastive_loss

    inmodal_cyclic_loss = torch.tensor(0).to(options.device)
    if options.cylambda1 > 0:
        logits_image_per_image = umodel.logit_scale.exp() * image_embeds @ image_embeds.t()
        logits_text_per_text = umodel.logit_scale.exp() * text_embeds @ text_embeds.t()
        inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (
                umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size

    crossmodal_cyclic_loss = torch.tensor(0).to(options.device)
    if options.cylambda2 > 0:
        crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (
                umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size

    cyclic_loss = options.cylambda1 * inmodal_cyclic_loss + options.cylambda2 * crossmodal_cyclic_loss
    loss = contrastive_loss + cyclic_loss

    return loss, contrastive_loss, cyclic_loss


def get_extra_loss(umodel, image_embeds, text_embeds, criterion, options):
    if options.distributed:
        gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
        gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]

        dist.all_gather(gathered_image_embeds, image_embeds)
        dist.all_gather(gathered_text_embeds, text_embeds)

        image_embeds = torch.cat(
            gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
        text_embeds = torch.cat(
            gathered_text_embeds[:options.rank] + [text_embeds] + gathered_text_embeds[options.rank + 1:])

    logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_image_per_text = logits_text_per_image.t()

    batch_size = len(logits_text_per_image)

    target = torch.arange(batch_size).long().to(options.device, non_blocking=True)

    crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
    contrastive_loss = crossmodal_contrastive_loss

    return contrastive_loss


def train(epoch, model, data, optimizer, scheduler, scaler, options):

    device = options.device

    if options.distributed:
        data['train'].set_epoch(epoch)
        data['train-count'].set_epoch(epoch)
        data['train-relative-size'].set_epoch(epoch)
        data['train-absolute-size'].set_epoch(epoch)
        data['train-relative-spatial'].set_epoch(epoch)
        data['train-absolute-spatial'].set_epoch(epoch)
        data['train-existence'].set_epoch(epoch)

    model.train()
    criterion = nn.CrossEntropyLoss().to(options.device)

    umodel = model.module if options.distributed else model

    for index in range(options.steps_per_epoch):
        start = time.time()
        step = options.steps_per_epoch * epoch + index
        scheduler(step)

        # >>> get data and resize, concatenate them >>>
        try:
            common_batch = next(data['train'].iterator)
        except:
            data['train'].iterator = iter(data['train'].dataloader)
            common_batch = next(data['train'].iterator)
        common_pixel_values = common_batch['pixel_values']
        common_attention_mask = common_batch['attention_mask']
        common_input_ids = common_batch['input_ids']
        n_common = len(common_pixel_values)

        try:
            count_batch = next(data['train-count'].iterator)
        except:
            data['train-count'].iterator = iter(data['train-count'].dataloader)
            count_batch = next(data['train-count'].iterator)
        count_pixel_values = rearrange(count_batch['pixel_values'], 'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        count_attention_mask = rearrange(count_batch['attention_mask'], 'BS_ins hard_num -> (BS_ins hard_num)')
        count_input_ids = rearrange(count_batch['input_ids'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_count = len(count_pixel_values)

        try:
            rel_size_batch = next(data['train-relative-size'].iterator)
        except:
            data['train-relative-size'].iterator = iter(data['train-relative-size'].dataloader)
            rel_size_batch = next(data['train-relative-size'].iterator)
        rel_size_pixel_values = rearrange(rel_size_batch['pixel_values'],
                                          'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        rel_size_attention_mask = rearrange(rel_size_batch['attention_mask'], 'BS_ins hard_num -> (BS_ins hard_num)')
        rel_size_input_ids = rearrange(rel_size_batch['input_ids'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_rel_size = len(rel_size_pixel_values)

        try:
            abs_size_batch = next(data['train-absolute-size'].iterator)
        except:
            data['train-absolute-size'].iterator = iter(data['train-absolute-size'].dataloader)
            abs_size_batch = next(data['train-absolute-size'].iterator)
        abs_size_pixel_values = rearrange(abs_size_batch['pixel_values'],
                                          'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        abs_size_attention_mask = rearrange(abs_size_batch['attention_mask'], 'BS_ins hard_num -> (BS_ins hard_num)')
        abs_size_input_ids = rearrange(abs_size_batch['input_ids'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_abs_size = len(abs_size_pixel_values)

        try:
            rel_spatial_batch = next(data['train-relative-spatial'].iterator)
        except:
            data['train-relative-spatial'].iterator = iter(data['train-relative-spatial'].dataloader)
            rel_spatial_batch = next(data['train-relative-spatial'].iterator)
        rel_spatial_pixel_values = rearrange(rel_spatial_batch['pixel_values'],
                                             'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        rel_spatial_attention_mask = rearrange(rel_spatial_batch['attention_mask'],
                                               'BS_ins hard_num -> (BS_ins hard_num)')
        rel_spatial_input_ids = rearrange(rel_spatial_batch['input_ids'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_rel_spatial = len(rel_spatial_pixel_values)

        try:
            abs_spatial_batch = next(data['train-absolute-spatial'].iterator)
        except:
            data['train-absolute-spatial'].iterator = iter(data['train-absolute-spatial'].dataloader)
            abs_spatial_batch = next(data['train-absolute-spatial'].iterator)
        abs_spatial_pixel_values = rearrange(abs_spatial_batch['pixel_values'],
                                             'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        abs_spatial_attention_mask = rearrange(abs_spatial_batch['attention_mask'],
                                               'BS_ins hard_num -> (BS_ins hard_num)')
        abs_spatial_input_ids = rearrange(abs_spatial_batch['input_ids'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_abs_spatial = len(abs_spatial_pixel_values)

        try:
            existence_batch = next(data['train-existence'].iterator)
        except:
            data['train-existence'].iterator = iter(data['train-existence'].dataloader)
            existence_batch = next(data['train-existence'].iterator)
        existence_pixel_values = rearrange(existence_batch['pixel_values'],
                                           'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        existence_attention_mask = rearrange(existence_batch['attention_mask'], 'BS_ins hard_num -> (BS_ins hard_num)')
        existence_input_ids = rearrange(existence_batch['input_ids'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_existence = len(existence_pixel_values)

        # concatenate data
        all_pixel_values = torch.cat([common_pixel_values, count_pixel_values, rel_size_pixel_values,
                                      abs_size_pixel_values, rel_spatial_pixel_values, abs_spatial_pixel_values,
                                      existence_pixel_values], dim=0).to(device=device, non_blocking=True)
        all_attention_mask = torch.cat([common_attention_mask, count_attention_mask, rel_size_attention_mask,
                                        abs_size_attention_mask, rel_spatial_attention_mask, abs_spatial_attention_mask,
                                        existence_attention_mask], dim=0).to(device=device, non_blocking=True)
        all_input_ids = torch.cat([common_input_ids, count_input_ids, rel_size_input_ids,
                                   abs_size_input_ids, rel_spatial_input_ids, abs_spatial_input_ids,
                                   existence_input_ids], dim=0).to(device=device, non_blocking=True)
        assert (n_common + n_count + n_rel_size + n_abs_size + n_rel_spatial + n_abs_spatial + n_existence) == len(
            all_pixel_values)

        # <<< get data and reshape, concatenate them <<<
        data_time = time.time() - start

        optimizer.zero_grad()

        # forward pass
        all_outputs = model(input_ids=all_input_ids, attention_mask=all_attention_mask, pixel_values=all_pixel_values)

        # separate outputs
        all_image_embeds = all_outputs.image_embeds
        all_text_embeds = all_outputs.text_embeds
        common_image_embeds = all_image_embeds[:n_common]
        common_text_embeds = all_text_embeds[:n_common]
        extra_image_embeds = all_image_embeds[n_common:]
        extra_text_embeds = all_text_embeds[n_common:]

        with autocast():
            # common loss
            common_loss, common_contrastive_loss, common_cyclic_loss = get_loss(umodel, common_image_embeds, common_text_embeds, criterion, options)
            # extra hard negative loss
            extra_loss = get_extra_loss(umodel, extra_image_embeds, extra_text_embeds, criterion, options)
            # total_loss
            loss = common_loss + options.extra_loss_wt * extra_loss
            # backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)

        scaler.update()
        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - start
        process_time = batch_time - data_time

        if options.master and (((index + 1) % options.log_per_steps == 0) or (index + 1 == options.steps_per_epoch)):
            logging.info(
                f"Train Epoch: {epoch + 1:02d} [{step}/{options.steps_per_epoch * options.epochs} ({100.0 * (index + 1) / options.steps_per_epoch:.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
                f"\tcomLoss: {common_loss.item():.6f}"
                f"\textLoss: {extra_loss.item():.6f}"
                f"\tBatch Time {batch_time:.2f}"
                f"\tData Time {data_time:.2f}"
                f"\tProcess Time {process_time:.2f}"
                f"\tScale {umodel.logit_scale.data:.4f}"
                f"\tLearning Rate: {optimizer.param_groups[0]['lr']:.10f}")

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time,
                "batch_time": batch_time,
                "process_time": process_time,
                "scale": umodel.logit_scale.data,
                "common_loss": common_loss.item(),
                "extra_loss": extra_loss.item(),
                "total_loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if options.wandb:
                log_data['step'] = step  # for backwards compatibility
                # wandb.log(log_data, step=step)
                wandb.log(log_data)
