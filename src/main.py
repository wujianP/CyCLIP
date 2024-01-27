import os

os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import wandb
import torch
import logging
import warnings
import numpy as np
import tensorflow as tf
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from pkgs.openai.clip import load as load_model

from .train import train
from .evaluate import evaluate
from .data import load as load_data
from .parser import parse_args
from .scheduler import cosine_scheduler
from .logger import get_logger, set_logger

mp.set_start_method("spawn", force=True)
warnings.filterwarnings("ignore")


def seconds_to_hms(t):
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)

    return f'{hours:02d} h, {minutes:02d} m, {seconds:02d} s'


def worker(rank, options, logger):
    options.rank = rank
    options.master = rank == 0

    set_logger(rank=rank, logger=logger, distributed=options.distributed)

    if options.device == "cuda":
        options.device += ":" + str(options.device_ids[options.rank] if options.distributed else options.device_id)

    logging.info(f"Using {options.device} device")

    if options.master:
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    if options.distributed:
        dist.init_process_group(backend=options.distributed_backend, init_method=options.distributed_init_method,
                                world_size=options.num_devices, rank=options.rank)

    # options.batch_size = options.batch_size // options.num_devices

    logging.info(f"Loading Model Begin")
    model, processor = load_model(name=options.model_name, pretrained=options.pretrained)
    logging.info(f"Loading Model Finished")

    if options.device == "cpu":
        model.float()
    else:
        torch.cuda.set_device(options.device_ids[options.rank] if options.distributed else options.device_id)
        model.to(options.device)
        if options.distributed:
            model = DDP(model, device_ids=[options.device_ids[options.rank]])

    logging.info(f"Loading Data Begin")
    data = load_data(options, processor)
    logging.info(f"Loading Data Finished")

    optimizer = None
    scheduler = None
    if data["train"] is not None:
        weight_decay_parameters = []
        no_weight_decay_parameters = []

        for name, parameter in model.named_parameters():
            if all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad:
                weight_decay_parameters.append(parameter)

            if any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad:
                no_weight_decay_parameters.append(parameter)

        optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0},
                                 {"params": weight_decay_parameters, "weight_decay": options.weight_decay}],
                                lr=options.lr, betas=(options.beta1, options.beta2), eps=options.eps)
        scheduler = cosine_scheduler(optimizer, options.lr, options.num_warmup_steps,
                                     options.steps_per_epoch * options.epochs)

    start_epoch = 0
    # load pretrained model weights
    if options.from_pretrained is not None:
        if os.path.isfile(options.from_pretrained):
            checkpoint = torch.load(options.from_pretrained, map_location=options.device)
            state_dict = checkpoint["state_dict"]
            if not options.distributed and next(iter(state_dict.items()))[0].startswith("module"):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")

    if options.checkpoint is not None:
        raise KeyError('Not support by wujian')
        # if os.path.isfile(options.checkpoint):
        #     checkpoint = torch.load(options.checkpoint, map_location=options.device)
        #     start_epoch = checkpoint["epoch"]
        #     state_dict = checkpoint["state_dict"]
        #     if not options.distributed and next(iter(state_dict.items()))[0].startswith("module"):
        #         state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        #     model.load_state_dict(state_dict)
        #     if optimizer is not None: optimizer.load_state_dict(checkpoint["optimizer"])
        #     logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
        # else:
        #     logging.info(f"No checkpoint found at {options.checkpoint}")

    cudnn.benchmark = True
    cudnn.deterministic = False

    if options.wandb and options.master:
        wandb_login_success = wandb.login(key=options.wandb_key, timeout=30)
        if not wandb_login_success:
            logging.info("Wandb init Failed!!!")
        else:
            logging.info("Starting wandb")
            wandb.init(project=options.wandb_project_name,
                       name=options.name,
                       id=options.name,
                       notes=options.notes,
                       tags=[],
                       resume=None,
                       config=vars(options))
            # wandb.run.name = options.name
            wandb.save(os.path.join(options.log_dir_path, "params.txt"))

    # training epoch by epoch
    if data["train"] is not None:
        options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(options.checkpoints_dir_path, exist_ok=True)

        scaler = GradScaler()

        for epoch in range(start_epoch, options.epochs):
            # >>> eval before training begin >>>
            if epoch == 0 and (not options.do_not_eval_epoch_0):
                if options.master:
                    logging.info(f'Start Evaluation before the : The Baseline Without Training')
                evaluate(epoch=epoch, model=model, processor=processor, options=options)
                torch.cuda.empty_cache()

            # >>> training one epoch >>>
            if options.master:
                logging.info(f"Starting Training Epoch {epoch + 1} / {options.epochs}")
            start = time.time()
            train(epoch, model, data, optimizer, scheduler, scaler, options)
            torch.cuda.empty_cache()
            end = time.time()
            if options.master:
                logging.info(f"Finished Training Epoch {epoch + 1} / {options.epochs}, Time Taken Per Epoch: {seconds_to_hms(end - start)}")

            # >>> eval one epoch >>>
            if (epoch + 1) % options.eval_per_epoch == 0 or (epoch + 1) == options.epochs:
                if options.master:
                    logging.info(f'Start Evaluating Epoch {epoch + 1} /  {options.epochs}')
                evaluate(epoch=epoch, model=model, processor=processor, options=options)
                torch.cuda.empty_cache()

            # >>> save checkpoints >>>
            if options.master:
                checkpoint = {"epoch": epoch + 1,
                              "name": options.name,
                              "state_dict": model.state_dict(),
                              "optimizer": optimizer.state_dict(),
                              "scaler": scaler.state_dict() if scaler is not None else None
                              }
                # period save
                if (epoch + 1) == options.epochs or ((epoch + 1) % options.save_per_epoch == 0):
                    torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch + 1}.pt"))
                # latest save
                if options.save_most_recent:
                    # try not to corrupt the latest checkpoint if save fails
                    tmp_save_path = os.path.join(options.checkpoints_dir_path, "tmp.pt")
                    latest_save_path = os.path.join(options.checkpoints_dir_path, "epoch_latest.pt")
                    torch.save(checkpoint, tmp_save_path)
                    os.replace(tmp_save_path, latest_save_path)

    if options.distributed:
        dist.destroy_process_group()

    if options.wandb and options.master:
        wandb.finish()


if __name__ == "__main__":
    options = parse_args()

    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")

    os.makedirs(options.log_dir_path, exist_ok=True)
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    ngpus = torch.cuda.device_count()
    if ngpus == 0 or options.device == "cpu":
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if ngpus == 1 or not options.distributed:
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if options.device_ids is None:
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs=options.num_devices, args=(options, logger))

    listener.stop()
