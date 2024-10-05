import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("/hkfs/home/project/hk-project-test-mlperf/om1434/masterarbeit")
from wind_fusion import energy_dataset

from era5_data import utils
from era5_data.config import cfg
import torch
import os
from torch.utils import data
from models.pangu_power_sample import test, train
from models.pangu_power import PanguPower
import argparse
import time
import logging
from tensorboardX import SummaryWriter

"""
Finetune pangu_power on the energy dataset
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_net", type=str, default="finetune_power")
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

    output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    utils.mkdirs(output_path)

    writer_path = os.path.join(output_path, "writer")
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)

    writer = SummaryWriter(writer_path)

    logger_name = args.type_net + str(cfg.PG.HORIZON)
    utils.logger_info(logger_name, os.path.join(output_path, logger_name + ".log"))

    logger = logging.getLogger(logger_name)

    train_dataset = energy_dataset.EnergyDataset(
        filepath_era5="/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr",
        filepath_power="/lsdf/kit/imk-tro/projects/Gruppe_Quinting/om1434/offshore/offshore.zarr",
        startDate="20170101",
        endDate="20171231",
        freq="24h",
    )

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

    val_dataset = energy_dataset.EnergyDataset(
        filepath_era5="/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr",
        filepath_power="/lsdf/kit/imk-tro/projects/Gruppe_Quinting/om1434/offshore/offshore.zarr",
        startDate="20180101",
        endDate="20180228",
        freq="24h",
    )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.PG.VAL.BATCH_SIZE,
        drop_last=True,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_dataset = energy_dataset.EnergyDataset(
        filepath_era5="/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr",
        filepath_power="/lsdf/kit/imk-tro/projects/Gruppe_Quinting/om1434/offshore/offshore.zarr",
        startDate="20180301",
        endDate="20180430",
        freq="24h",
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=cfg.PG.TEST.BATCH_SIZE,
        drop_last=True,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = PanguPower(device=device).to(device)

    checkpoint = torch.load(
        cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device, weights_only=False
    )

    pretrained_dict = checkpoint["model"]
    model_dict = model.state_dict()

    # Filter out keys in pretrained_dict that belong to _output_layer (conv and conv_surface)
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if "_output_layer" not in k
    }

    # Update the model's state_dict except the _output_layer
    model_dict.update(pretrained_dict)

    # Load the updated state_dict into the model
    model.load_state_dict(model_dict)

    # Only finetune the last layer
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "_output_layer" in name:
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.PG.TRAIN.LR,
        weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY,
    )

    msg = "\n"
    msg += utils.torch_summarize(model, show_weights=False)
    logger.info(msg)

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
