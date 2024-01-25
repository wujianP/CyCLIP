import time
import wandb
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
from einops import rearrange


def get_loss(umodel, outputs, criterion, options):
    if (options.inmodal):
        image_embeds, augmented_image_embeds = outputs.image_embeds[
                                               :len(outputs.image_embeds) // 2], outputs.image_embeds[
                                                                                 len(outputs.image_embeds) // 2:]
        text_embeds, augmented_text_embeds = outputs.text_embeds[:len(outputs.text_embeds) // 2], outputs.text_embeds[
                                                                                                  len(outputs.text_embeds) // 2:]
    else:
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    if (options.distributed):
        if (options.inmodal):
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
            augmented_gathered_image_embeds = [torch.zeros_like(augmented_image_embeds) for _ in
                                               range(options.num_devices)]
            augmented_gathered_text_embeds = [torch.zeros_like(augmented_text_embeds) for _ in
                                              range(options.num_devices)]

            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
            dist.all_gather(augmented_gathered_image_embeds, augmented_image_embeds)
            dist.all_gather(augmented_gathered_text_embeds, augmented_text_embeds)

            image_embeds = torch.cat(
                gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds = torch.cat(
                gathered_text_embeds[:options.rank] + [text_embeds] + gathered_text_embeds[options.rank + 1:])
            augmented_image_embeds = torch.cat(augmented_gathered_image_embeds[:options.rank] + [
                augmented_image_embeds] + augmented_gathered_image_embeds[options.rank + 1:])
            augmented_text_embeds = torch.cat(augmented_gathered_text_embeds[:options.rank] + [
                augmented_text_embeds] + augmented_gathered_text_embeds[options.rank + 1:])
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

    if (options.inmodal):
        logits_image_per_augmented_image = umodel.logit_scale.exp() * image_embeds @ augmented_image_embeds.t()
        logits_text_per_augmented_text = umodel.logit_scale.exp() * text_embeds @ augmented_text_embeds.t()

    batch_size = len(logits_text_per_image)

    target = torch.arange(batch_size).long().to(options.device, non_blocking=True)

    contrastive_loss = torch.tensor(0).to(options.device)
    if (options.inmodal):
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text,
                                                                                            target)) / 2
        inmodal_contrastive_loss = (criterion(logits_image_per_augmented_image, target) + criterion(
            logits_text_per_augmented_text, target)) / 2
        contrastive_loss = (crossmodal_contrastive_loss + inmodal_contrastive_loss) / 2
    else:
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text,
                                                                                            target)) / 2
        contrastive_loss = crossmodal_contrastive_loss

    inmodal_cyclic_loss = torch.tensor(0).to(options.device)
    if (options.cylambda1 > 0):
        logits_image_per_image = umodel.logit_scale.exp() * image_embeds @ image_embeds.t()
        logits_text_per_text = umodel.logit_scale.exp() * text_embeds @ text_embeds.t()
        inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (
                    umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size

    crossmodal_cyclic_loss = torch.tensor(0).to(options.device)
    if (options.cylambda2 > 0):
        crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (
                    umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size

    cyclic_loss = options.cylambda1 * inmodal_cyclic_loss + options.cylambda2 * crossmodal_cyclic_loss
    loss = contrastive_loss + cyclic_loss

    return loss, contrastive_loss, cyclic_loss


def train(epoch, model, data, optimizer, scheduler, scaler, options):
    from IPython import embed
    embed(header='train')

    dataloader = data["train"]

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

    modulo = max(1, int(dataloader.num_samples / options.batch_size / 10))
    umodel = model.module if options.distributed else model

    start = time.time()

    # logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}")
    # for index, batch in enumerate(dataloader):
    for index in range(options.steps_per_epoch):
        step = options.steps_per_epoch * epoch + index
        scheduler(step)

        # >>> get data and resize them >>>
        try:
            common_batch = next(data['train'].iterator)
        except:
            data['train'].iterator = iter(data['train'].dataloader)
            common_batch = next(data['train'].iterator)
        common_images = common_batch[0]
        common_texts = common_batch[1]
        n_common = len(common_images)

        try:
            count_batch = next(data['train-count'].iterator)
        except:
            data['train-count'].iterator = iter(data['train-count'].dataloader)
            count_batch = next(data['train-count'].iterator)
        count_images = rearrange(count_batch['images'], 'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        count_texts = rearrange(count_batch['texts'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_count = len(count_images)

        try:
            rel_size_batch = next(data['train-relative-size'].iterator)
        except:
            data['train-relative-size'].iterator = iter(data['train-relative-size'].dataloader)
            rel_size_batch = next(data['train-relative-size'].iterator)
        rel_size_images = rearrange(rel_size_batch['images'], 'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        rel_size_texts = rearrange(rel_size_batch['texts'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_rel_size = len(rel_size_images)

        try:
            abs_size_batch = next(data['train-absolute-size'].iterator)
        except:
            data['train-absolute-size'].iterator = iter(data['train-absolute-size'].dataloader)
            abs_size_batch = next(data['train-absolute-size'].iterator)
        abs_size_images = rearrange(abs_size_batch['images'], 'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        abs_size_texts = rearrange(abs_size_batch['texts'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_abs_size = len(abs_size_images)

        try:
            rel_spatial_batch = next(data['train-relative-spatial'].iterator)
        except:
            data['train-relative-spatial'].iterator = iter(data['train-relative-spatial'].dataloader)
            rel_spatial_batch = next(data['train-relative-spatial'].iterator)
        rel_spatial_images = rearrange(rel_spatial_batch['images'], 'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        rel_spatial_texts = rearrange(rel_spatial_batch['texts'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_rel_spatial = len(rel_spatial_images)

        try:
            abs_spatial_batch = next(data['train-absolute-spatial'].iterator)
        except:
            data['train-absolute-spatial'].iterator = iter(data['train-absolute-spatial'].dataloader)
            abs_spatial_batch = next(data['train-absolute-spatial'].iterator)
        abs_spatial_images = rearrange(abs_spatial_batch['images'], 'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        abs_spatial_texts = rearrange(abs_spatial_batch['texts'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_abs_spatial = len(abs_spatial_images)

        try:
            existence_batch = next(data['train-existence'].iterator)
        except:
            data['train-existence'].iterator = iter(data['train-existence'].dataloader)
            existence_batch = next(data['train-existence'].iterator)
        existence_images = rearrange(existence_batch['images'], 'BS_ins hard_num C W H -> (BS_ins hard_num) C W H')
        existence_texts = rearrange(existence_batch['texts'], 'BS_ins hard_num L -> (BS_ins hard_num) L')
        n_existence = len(existence_images)
        # <<< get data and reshape them <<<

        optimizer.zero_grad()

        if options.inmodal:
            raise KeyError
            # input_ids, attention_mask, pixel_values = batch["input_ids"][0].to(options.device, non_blocking=True), \
            # batch["attention_mask"][0].to(options.device, non_blocking=True), batch["pixel_values"][0].to(
            #     options.device, non_blocking=True)
            # augmented_input_ids, augmented_attention_mask, augmented_pixel_values = batch["input_ids"][1].to(
            #     options.device, non_blocking=True), batch["attention_mask"][1].to(options.device, non_blocking=True), \
            # batch["pixel_values"][1].to(options.device, non_blocking=True)
            # input_ids = torch.cat([input_ids, augmented_input_ids])
            # attention_mask = torch.cat([attention_mask, augmented_attention_mask])
            # pixel_values = torch.cat([pixel_values, augmented_pixel_values])
        else:
            input_ids, attention_mask, pixel_values = common_batch["input_ids"].to(options.device, non_blocking=True), common_batch[
                "attention_mask"].to(options.device, non_blocking=True), common_batch["pixel_values"].to(options.device,
                                                                                                  non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        with autocast():
            loss, contrastive_loss, cyclic_loss = get_loss(umodel, outputs, criterion, options)
            scaler.scale(loss).backward()
            scaler.step(optimizer)

        scaler.update()
        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)

        end = time.time()

        if options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1)):
            num_samples = (index + 1) * len(input_ids) * options.num_devices
            dataloader_num_samples = dataloader.num_samples

            logging.info(
                f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader_num_samples} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")

            metrics = {"loss": loss.item(), "contrastive_loss": contrastive_loss.item(),
                       "cyclic_loss": cyclic_loss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
            if options.wandb:
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step})

            start = time.time()
