import os
import copy
import torch
from torch import nn
import torch.distributed as dist
from datetime import datetime
import warnings
from era5_data import utils, utils_data
from era5_data.config import cfg
from typing import Tuple, Dict, List, Union
import logging
from tensorboardX import SummaryWriter


warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
)

warnings.filterwarnings(
    "ignore",
    message="Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas",
)


def load_constants(device):
    return utils_data.loadAllConstants(device=device)


def load_land_sea_mask(device, mask_type="sea", fill_value=0):
    return utils_data.loadLandSeaMask(
        device, mask_type=mask_type, fill_value=fill_value
    )


def model_inference(
    model: nn.Module,
    input: torch.Tensor,
    input_surface: torch.Tensor,
    aux_constants: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    output_power, output_surface = model(
        input,
        input_surface,
        aux_constants["weather_statistics"],
        aux_constants["constant_maps"],
        aux_constants["const_h"],
    )

    # Transfer to the output to the original data range
    output_surface = utils_data.normBackDataSurface(
        output_surface, aux_constants["weather_statistics_last"]
    )

    return output_power, output_surface


def calculate_loss(output, target, criterion, lsm_expanded):
    mask_not_zero = ~(lsm_expanded == 0)
    mask_not_zero = mask_not_zero.unsqueeze(1)
    output = output * lsm_expanded
    loss = criterion(output[mask_not_zero], target[mask_not_zero])
    return torch.mean(loss)


def visualize(
    output_power,
    target_power,
    input_surface,
    output_surface,
    target_surface,
    step,
    path,
):
    utils.visuailze_all(
        output_power.detach().cpu().squeeze(),
        target_power.detach().cpu().squeeze(),
        input_surface.detach().cpu().squeeze(),
        output_surface.detach().cpu().squeeze(),
        target_surface.detach().cpu().squeeze(),
        step=step,
        path=path,
    )


def save_output_and_target(output_test, target_test, target_time, res_path):
    output_path = os.path.join(res_path, "model_output")
    utils.mkdirs(output_path)
    torch.save(output_test, os.path.join(output_path, f"output_{target_time}.pth"))
    torch.save(target_test, os.path.join(output_path, f"target_{target_time}.pth"))


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    res_path: str,
    writer: SummaryWriter,
    logger: logging.Logger,
    start_epoch: int,
    rank: int = 0,
    device: Union[torch.device, None] = None,
) -> nn.Module:
    """Training code"""
    criterion = nn.L1Loss(reduction="none")
    epochs = cfg.PG.TRAIN.EPOCHS
    loss_list: List[float] = []
    best_loss = float("inf")
    epochs_since_last_improvement = 0
    best_model = model
    aux_constants = load_constants(device)

    # Termination flag to signal early stopping
    early_stop_flag = torch.tensor(
        [0], dtype=torch.int, device=device
    )  # 0 means continue, 1 means stop

    for i in range(start_epoch, epochs + 1):
        if early_stop_flag.item() == 1:
            break  # If early stop flag is set, break out of the loop

        epoch_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            aux_constants,
            logger,
            rank,
            device,
            i,
        )
        loss_list.append(epoch_loss)
        lr_scheduler.step()

        if rank == 0 and i % cfg.PG.TRAIN.SAVE_INTERVAL == 0:
            save_model_checkpoint(model, optimizer, lr_scheduler, res_path, i)

        # Validate on all ranks (on purpose, since barrier times out if only rank 0 validates)
        # TODO(EliasKng): Use DistributedSampler to spread validation across all ranks, then calculate mean loss
        if i % cfg.PG.VAL.INTERVAL == 0:
            val_loss, best_model, epochs_since_last_improvement = validate(
                model,
                val_loader,
                criterion,
                aux_constants,
                writer,
                logger,
                res_path,
                best_loss,
                epoch_loss,
                best_model,
                epochs_since_last_improvement,
                rank,
                device,
                i,
            )

            # DO NOT COMMIT
            if rank == 0:
                early_stop_flag[0] = 1

            # Set early stop flag
            if rank == 0 and epochs_since_last_improvement >= 5:
                logger.info(
                    f"No improvement in validation loss for {epochs_since_last_improvement} epochs, terminating training."
                )
                early_stop_flag[0] = 1  # Set the early stop flag

        # Broadcast early stop flag from rank 0 to all other ranks
        dist.broadcast(early_stop_flag, src=0)

        # Synchronize all ranks
        dist.barrier()

    return best_model


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    aux_constants: Dict[str, torch.Tensor],
    logger: logging.Logger,
    rank: int,
    device: Union[torch.device, None],
    epoch: int,
) -> float:
    epoch_loss = 0.0
    print(f"Starting epoch {epoch}/{cfg.PG.TRAIN.EPOCHS}")

    for id, train_data in enumerate(train_loader):
        (
            input,
            input_surface,
            target_power,
            target_upper,
            target_surface,
            periods,
        ) = train_data
        input, input_surface, target_power = (
            input.to(device),
            input_surface.to(device),
            target_power.to(device),
        )
        print(f"(T) Processing batch {id + 1}/{len(train_loader)}")

        optimizer.zero_grad()
        model.train()
        output_power, output_surface = model_inference(
            model, input, input_surface, aux_constants
        )
        lsm_expanded = load_land_sea_mask(output_power.device)
        loss = calculate_loss(output_power, target_power, criterion, lsm_expanded)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f"Epoch {epoch} finished with training loss: {epoch_loss:.4f}")
    if rank == 0:
        logger.info("Epoch {} : {:.3f}".format(epoch, epoch_loss))

    return epoch_loss


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    res_path: str,
    epoch: int,
) -> None:
    model_save_path = os.path.join(res_path, "models")
    utils.mkdirs(model_save_path)
    save_file = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(save_file, os.path.join(model_save_path, "train_{}.pth".format(epoch)))
    print("Model saved at epoch {}".format(epoch))


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    aux_constants: Dict[str, torch.Tensor],
    writer: SummaryWriter,
    logger: logging.Logger,
    res_path: str,
    best_loss: float,
    epoch_loss: float,
    best_model: nn.Module,
    epochs_since_last_improvement: int,
    rank: int,
    device: Union[torch.device, None],
    epoch: int,
) -> Tuple[float, nn.Module, int]:
    print(f"Starting validation at epoch {epoch}")
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        for id, val_data in enumerate(val_loader, 0):
            (
                input_val,
                input_surface_val,
                target_power_val,
                target_upper_val,
                target_surface_val,
                periods_val,
            ) = val_data
            input_val, input_surface_val, target_power_val = (
                input_val.to(device),
                input_surface_val.to(device),
                target_power_val.to(device),
            )
            print(f"(V) Processing batch {id + 1}/{len(val_loader)}")
            output_power_val, output_surface_val = model_inference(
                model, input_val, input_surface_val, aux_constants
            )
            lsm_expanded = load_land_sea_mask(output_power_val.device)
            loss = calculate_loss(
                output_power_val, target_power_val, criterion, lsm_expanded
            )
            val_loss += loss.item()

        val_loss /= len(val_loader)
        if rank == 0:
            writer.add_scalars("Loss", {"train": epoch_loss, "val": val_loss}, epoch)
            logger.info("Validate at Epoch {} : {:.3f}".format(epoch, val_loss))
            png_path = os.path.join(res_path, "png_training")
            utils.mkdirs(png_path)
            visualize(
                output_power_val,
                target_power_val,
                input_surface_val,
                output_surface_val,
                target_surface_val,
                epoch,
                png_path,
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.module)
            if rank == 0:
                torch.save(
                    best_model, os.path.join(res_path, "models", "best_model.pth")
                )
                logger.info(
                    f"New best model saved at epoch {epoch} with validation loss: {val_loss:.4f}"
                )
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1

    return val_loss, best_model, epochs_since_last_improvement


def test(test_loader, model, device, res_path):
    aux_constants = load_constants(device)
    for id, data in enumerate(test_loader, 0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] predict on {id}")
        (
            input_test,
            input_surface_test,
            target_power_test,
            target_upper_test,
            target_surface_test,
            periods_test,
        ) = data

        input_test, input_surface_test, target_power_test = (
            input_test.to(device),
            input_surface_test.to(device),
            target_power_test.to(device),
        )
        model.eval()
        output_power_test, output_surface_test = model_inference(
            model, input_test, input_surface_test, aux_constants
        )
        lsm_expanded = load_land_sea_mask(output_power_test.device)
        output_power_test = output_power_test * lsm_expanded
        target_time = periods_test[1][0]
        png_path = os.path.join(res_path, "png")
        utils.mkdirs(png_path)
        visualize(
            output_power_test,
            target_power_test,
            input_surface_test,
            output_surface_test,
            target_surface_test,
            target_time,
            png_path,
        )
