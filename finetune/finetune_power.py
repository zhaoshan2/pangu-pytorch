import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("/hkfs/home/project/hk-project-test-mlperf/om1434/masterarbeit")
from wind_fusion import energy_dataset
from era5_data import utils
from era5_data.config import cfg
import torch
from torch.optim.adam import Adam
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils import data
from models.pangu_power_sample import test, train
from models.pangu_power import (
    PanguPowerPatchRecovery,
    PanguPowerConv,
    PanguPowerConvSigmoid,
)
import argparse
import logging
from tensorboardX import SummaryWriter

"""
Finetune pangu_power on the energy dataset
"""


def setup_model(model_type: str, device: torch.device) -> torch.nn.Module:
    """Loads the specified model and sets requires_grad

    Parameters
    ----------
    model_type : str
        Which model to load
    """
    if model_type == "PanguPowerPatchRecovery":
        model = PanguPowerPatchRecovery(device=device).to(device)
        model.load_pangu_state_dict(device)

        # Only finetune the last layer
        set_requires_grad(model, "_output_power_layer")

    elif model_type == "PanguPowerConv":
        model = PanguPowerConv(device=device).to(device)
        checkpoint = torch.load(
            "/home/hk-project-test-mlperf/om1434/masterarbeit/wind_fusion/pangu_pytorch/result/PanguPowerConv_64_128_64_1_k3_2/24/models/train_6.pth",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])

        # Only finetune the last layer
        set_requires_grad(model, "_conv_power_layers")

    elif model_type == "PanguPowerConvSigmoid":
        model = PanguPowerConvSigmoid(device=device).to(device)
        checkpoint = torch.load(
            cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"], strict=False)
        set_requires_grad(model, "_conv_power_layers")

    else:
        raise ValueError("Model not found")

    return model


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
    os.environ["MASTER_PORT"] = "12357"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def create_dataloader(
    start: str,
    end: str,
    freq: str,
    batch_size: int,
    shuffle: bool,
    distributed: bool = False,
) -> data.DataLoader:
    dataset = energy_dataset.EnergyDataset(
        filepath_era5=cfg.ERA5_PATH,
        filepath_power=cfg.POWER_PATH,
        startDate=start,
        endDate=end,
        freq=freq,
    )
    if not distributed:
        return data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )
    train_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        sampler=train_sampler,
    )


def set_requires_grad(model: torch.nn.Module, layer_name: str) -> None:
    """
    Sets the `requires_grad` attribute of the parameters in the model.
    This function will first set `requires_grad` to False for all parameters in the model.
    Then, it will set `requires_grad` to True for all parameters whose names contain the specified `layer_name`.
    Parameters
    ----------
    model : torch.nn.Module
        The neural network model whose parameters' `requires_grad` attribute will be modified.
    layer_name : str
        The name (or partial name) of the layer whose parameters should have `requires_grad` set to True.
    """

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True
            print("Requires grad: ", name)


def setup_writer(output_path: str) -> SummaryWriter:
    writer_path = os.path.join(output_path, "writer")
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    writer = SummaryWriter(writer_path)
    return writer


def setup_logger(type_net: str, horizon: int, output_path: str) -> logging.Logger:
    logger_name = type_net + str(horizon)
    utils.logger_info(logger_name, os.path.join(output_path, logger_name + ".log"))
    logger = logging.getLogger(logger_name)
    return logger


def _get_device(rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def main(rank: int, args: argparse.Namespace, world_size: int) -> None:
    ddp_setup(rank, world_size)

    print(f"Rank: {rank}, World Size: {world_size}")

    output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    utils.mkdirs(output_path)

    writer = setup_writer(output_path)
    logger = setup_logger(args.type_net, cfg.PG.HORIZON, output_path)

    device = _get_device(rank)

    if rank == 0:
        logger.info(f"Start finetuning {args.type_net} on energy dataset")

    train_dataloader = create_dataloader(
        cfg.PG.TRAIN.START_TIME,
        cfg.PG.TRAIN.END_TIME,
        cfg.PG.TRAIN.FREQUENCY,
        cfg.PG.TRAIN.BATCH_SIZE,
        True,
        args.dist,
    )
    val_dataloader = create_dataloader(
        cfg.PG.VAL.START_TIME,
        cfg.PG.VAL.END_TIME,
        cfg.PG.VAL.FREQUENCY,
        cfg.PG.VAL.BATCH_SIZE,
        False,
    )

    model = setup_model("PanguPowerConv", device)
    model = DDP(model, device_ids=[device])

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.PG.TRAIN.LR,
        weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY,
    )
    if rank == 0:
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
        writer=writer,
        logger=logger,
        start_epoch=start_epoch,
        rank=rank,
        device=device,
    )

    destroy_process_group()


def test_best_model(args):
    output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    utils.mkdirs(output_path)
    logger = setup_logger(args.type_net, cfg.PG.HORIZON, output_path)
    logger.info("Begin testing...")
    device = _get_device(0)

    if args.load_my_best:
        best_model = torch.load(
            os.path.join(output_path, "models/best_model.pth"),
            map_location=device,
            weights_only=False,
        )

    test_dataloader = create_dataloader(
        cfg.PG.TEST.START_TIME,
        cfg.PG.TEST.END_TIME,
        cfg.PG.TEST.FREQUENCY,
        cfg.PG.TEST.BATCH_SIZE,
        False,
    )

    test(
        test_loader=test_dataloader,
        model=best_model,
        device=device,
        res_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_net", type=str, default="PatchRecoveryAllDist")
    parser.add_argument("--load_my_best", type=bool, default=True)
    parser.add_argument("--launcher", default="pytorch", help="job launcher")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--dist", default=True)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    print(f"World size: {world_size}")

    if args.dist and torch.cuda.is_available():
        mp.spawn(main, args=(args, world_size), nprocs=world_size)
    else:
        main(0, args, 1)
    test_best_model(args)
