import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from era5_data import utils, utils_data
from era5_data.utils_dist import get_dist_info, init_dist
from era5_data.config import cfg
from models.pangu_model import PanguModel
import torch
import os
from torch.utils import data
from models.pangu_sample import test, train
import argparse
import time
import logging
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

"""
Fully finetune the pretrained model
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_net", type=str, default="finetune_fully")
    parser.add_argument("--load_my_best", type=bool, default=True)
    parser.add_argument("--launcher", default="pytorch", help="job launcher")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--dist", default=True)

    args = parser.parse_args()
    starts = time.time()

    PATH = cfg.PG_INPUT_PATH

    opt = {"gpu_ids": [0, 1, 2, 3]}
    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    # gpu_list = str(opt['gpu_ids'])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    device = torch.device("cuda" if opt["gpu_ids"] else "cpu")

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    print("Setup distributed settings")
    if args.dist:
        init_dist("pytorch")
    rank, world_size = get_dist_info()
    print("The rank and world size is", rank, world_size)
    if rank == 0:
        print(f"Predicting on {device}")

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
    if rank == 0:
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

    checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch)
    model.load_state_dict(checkpoint["model"])
    # Fully finetune
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.PG.TRAIN.LR,
        weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY,
    )

    if rank == 0:
        msg = "\n"
        msg += utils.torch_summarize(model, show_weights=False)
        logger.info(msg)

    # weather_statistics = utils.LoadStatic_pretrain()
    if rank == 0:
        print("weather statistics are loaded!")
    torch.set_num_threads(cfg.GLOBAL.NUM_STREADS)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 50], gamma=0.5
    )
    start_epoch = 1

    model = train(
        model,
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

    if args.load_my_best:
        best_model = torch.load(
            os.path.join(output_path, "models/best_model.pth"), map_location="cuda:0"
        )

    logger.info("Begin testing...")

    test(
        test_loader=test_dataloader,
        model=best_model,
        device=device,
        res_path=output_path,
    )
    # CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 finetune_lastLayer_ddp.py --dist True
