import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from era5_data import utils, utils_data
from era5_data.utils_dist import get_dist_info, init_dist
from era5_data.config import cfg
from models.pangu_model import PanguModel
from torch import nn
import torch
from torch.utils import data
from torch.optim.adam import Adam
from models.pangu_sample import test, train
import argparse
import time
import logging
import copy
import warnings
from tensorboardX import SummaryWriter
from peft import LoraConfig, get_peft_model
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings("ignore", category=FutureWarning)

"""
Finetune the model using parameter-efficient finetune (lora)
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_net", type=str, default="loratuner_normout_64")
    parser.add_argument("--load_pretrained", type=bool, default=False)
    parser.add_argument("--load_my_best", type=bool, default=True)
    parser.add_argument("--launcher", default="pytorch", help="job launcher")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--dist", default=False)
    args = parser.parse_args()
    starts = time.time()

    PATH = cfg.PG_INPUT_PATH

    opt = {
        "gpu_ids": list(range(torch.cuda.device_count()))
    }  # Automatically select available GPUs
    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    print(f"Available GPUs: {gpu_list}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Predicting on {device}")
    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if args.dist:
        init_dist("pytorch")
        rank, world_size = get_dist_info()
        print("The rank and world size is", rank, world_size)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Predicting on {device}")

    output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    utils.mkdirs(output_path)

    writer_path = os.path.join(output_path, "writer")
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)

    writer = SummaryWriter(writer_path)

    logger_name = args.type_net + str(cfg.PG.HORIZON)
    utils.logger_info(logger_name, os.path.join(output_path, logger_name + ".log"))

    logger = logging.getLogger(logger_name)

    train_dataset = utils_data.NetCDFDataset(
        nc_path=PATH,
        data_transform=None,
        training=True,
        validation=False,
        startDate=cfg.PG.TRAIN.START_TIME,
        endDate=cfg.PG.TRAIN.END_TIME,
        freq=cfg.PG.TRAIN.FREQUENCY,
        horizon=cfg.PG.HORIZON,
    )
    if args.dist:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)

        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=cfg.PG.TRAIN.BATCH_SIZE,
            num_workers=0,
            pin_memory=False,
            sampler=train_sampler,
        )
    else:
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=cfg.PG.TRAIN.BATCH_SIZE,
            drop_last=True,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    dataset_length = len(train_dataloader)
    print("dataset_length", dataset_length)

    val_dataset = utils_data.NetCDFDataset(
        nc_path=PATH,
        data_transform=None,
        training=False,
        validation=True,
        startDate=cfg.PG.VAL.START_TIME,
        endDate=cfg.PG.VAL.END_TIME,
        freq=cfg.PG.VAL.FREQUENCY,
        horizon=cfg.PG.HORIZON,
    )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.PG.VAL.BATCH_SIZE,
        drop_last=True,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_dataset = utils_data.NetCDFDataset(
        nc_path=PATH,
        data_transform=None,
        training=False,
        validation=False,
        startDate=cfg.PG.TEST.START_TIME,
        endDate=cfg.PG.TEST.END_TIME,
        freq=cfg.PG.TEST.FREQUENCY,
        horizon=cfg.PG.HORIZON,
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=cfg.PG.TEST.BATCH_SIZE,
        drop_last=True,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = PanguModel(device=device).to(device)
    checkpoint = torch.load(
        cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device, weights_only=True
    )
    print("loading model pretrained weight.")
    model.load_state_dict(checkpoint["model"])

    print([(n, type(m)) for n, m in model.named_modules()])
    target_modules = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            target_modules.append(n)
            print(f"appended {n}")
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        modules_to_save=["_output_layer.conv_surface", "_output_layer.conv"],
    )

    module_copy = copy.deepcopy(model)  # we keep a copy of the original model for later

    peft_model = get_peft_model(model, config)
    optimizer = Adam(
        peft_model.parameters(),
        lr=cfg.PG.TRAIN.LR,
        weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 50], gamma=0.5
    )
    start_epoch = 1
    if args.load_pretrained:
        cpk = torch.load(os.path.join(output_path, "models/train_30.pth"))
        peft_model.load_state_dict(cpk["model"])
        optimizer.load_state_dict(cpk["optimizer"])
        lr_scheduler.load_state_dict(cpk["lr_scheduler"])
        start_epoch = cpk["epoch"]

    peft_model = train(
        peft_model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        res_path=output_path,
        device=device,
        writer=writer,
        logger=logger,
        start_epoch=start_epoch,
    )

    for name, param in peft_model.base_model.named_parameters():
        if "lora" not in name:
            continue

        print(f"New parameter {name:<13} | {param.numel():>5} parameters | updated")

    params_before = dict(module_copy.named_parameters())
    for name, param in peft_model.base_model.named_parameters():
        if "lora" in name:
            continue

        name_before = (
            name.partition(".")[-1]
            .replace("original_", "")
            .replace("module.", "")
            .replace("modules_to_save.default.", "")
        )
        param_before = params_before[name_before]
        if torch.allclose(param, param_before):
            print(
                f"Parameter {name_before:<13} | {param.numel():>7} parameters | not updated"
            )
        else:
            print(
                f"Parameter {name_before:<13} | {param.numel():>7} parameters | updated"
            )

    output_path = os.path.join(output_path, "test")
    utils.mkdirs(output_path)

    test(
        test_loader=test_dataloader,
        model=peft_model,
        device=device,
        res_path=output_path,
    )
