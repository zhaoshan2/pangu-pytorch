import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from era5_data import utils, utils_data
from era5_data.config import cfg
from torch import nn
import torch
import copy
from datetime import datetime
import warnings


warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    res_path,
    device,
    writer,
    logger,
    start_epoch,
    rank=0,
):
    """Training code"""
    # Prepare for the optimizer and scheduler
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=- 1, verbose=False) #used in the paper

    # Loss function
    criterion = nn.L1Loss(reduction="none")

    # training epoch
    epochs = cfg.PG.TRAIN.EPOCHS

    loss_list = []
    best_loss = float("inf")
    epochs_since_last_improvement = 0
    best_model = None
    # scaler = torch.cuda.amp.GradScaler()

    # Load constants and teleconnection indices
    aux_constants = utils_data.loadAllConstants(
        device=device
    )  # 'weather_statistics','weather_statistics_last','constant_maps','tele_indices','variable_weights'
    upper_weights, surface_weights = aux_constants["variable_weights"]

    # Train a single Pangu-Weather model
    for i in range(start_epoch, epochs + 1):
        epoch_loss = 0.0
        print(f"Starting epoch {i}/{epochs}")

        for id, train_data in enumerate(train_loader):
            # Load weather data at time t as the input; load weather data at time t+336 as the output
            # Note the data need to be randomly shuffled
            input, input_surface, target, periods = train_data
            input, input_surface, target = (
                input.to(device),
                input_surface.to(device),
                target.to(device),
            )
            print(f"(T) Processing batch {id + 1}/{len(train_loader)}")

            optimizer.zero_grad()
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            # /with torch.cuda.amp.autocast():
            model.train()

            # Note the input and target need to be normalized (done within the function)
            # Call the model and get the output
            output = model(
                input,
                input_surface,
                aux_constants["weather_statistics"],
                aux_constants["constant_maps"],
                aux_constants["const_h"],
            )  # (1,5,13,721,1440) & (1, 4, 721, 1440)

            if cfg.PG.TRAIN.USE_LSM:
                device_upper = output.device
                lsm_expanded = utils_data.loadLandSeaMask(
                    device_upper, mask_type="sea", fill_value=0
                )

                # Multiply output_test with the land-sea mask
                output = output * lsm_expanded

            # We use the MAE loss to train the model
            loss = criterion(output, target)  # [1, 1, 721, 1440]

            # Truncate loss to EU area
            loss = loss[:, :, 721 - 651 : 721 - 466, :183]

            # Call the backward algorithm and calculate the gratitude of parameters
            # scaler.scale(loss).backward()
            loss = torch.mean(loss)
            loss.backward()

            # Update model parameters with Adam optimizer
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            epoch_loss += loss.item()

        print(
            f"Epoch {i} finished with training loss: {epoch_loss / len(train_loader):.4f}"
        )
        epoch_loss /= len(train_loader)
        if rank == 0:
            logger.info("Epoch {} : {:.3f}".format(i, epoch_loss))
        loss_list.append(epoch_loss)
        lr_scheduler.step()
        # scaler.update(lr_scheduler)
        #
        # for name, param in model.named_parameters():
        #   writer.add_histogram(name, param.data, i)

        model_save_path = os.path.join(res_path, "models")
        utils.mkdirs(model_save_path)

        # Save the training model
        if i % cfg.PG.TRAIN.SAVE_INTERVAL == 0:
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": i,
            }
            torch.save(
                save_file, os.path.join(model_save_path, "train_{}.pth".format(i))
            )
            # torch.save(model, os.path.join(model_save_path,'train_{}.pth'.format(i)))

            print("Model saved at epoch {}".format(i))
            print(
                "Save path: ", os.path.join(model_save_path, "train_{}.pth".format(i))
            )

        # TODO(EliasKng): Adapt validation code
        # Begin to validate
        if i % cfg.PG.VAL.INTERVAL == 0:
            print(f"Starting validation at epoch {i}")
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                for id, val_data in enumerate(val_loader, 0):
                    (
                        input_val,
                        input_surface_val,
                        target_val,
                        periods_val,
                    ) = val_data
                    input_val, input_surface_val, target_val = (
                        input_val.to(device),
                        input_surface_val.to(device),
                        target_val.to(device),
                    )

                    print(f"(V) Processing batch {id + 1}/{len(val_loader)}")

                    # Inference
                    output_val = model(
                        input_val,
                        input_surface_val,
                        aux_constants["weather_statistics"],
                        aux_constants["constant_maps"],
                        aux_constants["const_h"],
                    )

                    if cfg.PG.TRAIN.USE_LSM:
                        device_val = output_val.device
                        lsm_expanded = utils_data.loadLandSeaMask(
                            device_val, mask_type="sea", fill_value=0
                        )

                        # Multiply output_test with the land-sea mask
                        output_val = output_val * lsm_expanded

                    loss = criterion(output_val, target_val)

                    # Truncate loss to EU area
                    loss = loss[:, :, 721 - 651 : 721 - 466, :183]
                    loss = torch.mean(loss)
                    val_loss += loss.item()

                val_loss /= len(val_loader)
                print(f"Validation loss at epoch {i}: {val_loss:.4f}")
                writer.add_scalars("Loss", {"train": epoch_loss, "val": val_loss}, i)
                logger.info("Validate at Epoch {} : {:.3f}".format(i, val_loss))
                # Visualize the training process
                png_path = os.path.join(res_path, "png_training")
                utils.mkdirs(png_path)

                # utils.visuailze(
                #     output_val.detach().cpu().squeeze(),
                #     target_val.detach().cpu().squeeze(),
                #     input_val_raw.squeeze(),
                #     var="u",
                #     z=12,
                #     step=i,
                #     path=png_path,
                # )
                # utils.visuailze_surface(
                #     output_surface_val.detach().cpu().squeeze(),
                #     target_surface_val.detach().cpu().squeeze(),
                #     input_surface_val_raw.squeeze(),
                #     var="msl",
                #     step=i,
                #     path=png_path,
                # )

                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(model)
                    # Save the best model
                    torch.save(
                        best_model, os.path.join(model_save_path, "best_model.pth")
                    )
                    print(
                        f"New best model saved at epoch {i} with validation loss: {val_loss:.4f}"
                    )
                    logger.info(f"current best model is saved at {i} epoch.")
                    epochs_since_last_improvement = 0
                else:
                    epochs_since_last_improvement += 1
                    if epochs_since_last_improvement >= 5:
                        print(
                            f"No improvement in validation loss for {epochs_since_last_improvement} epochs, terminating training."
                        )
                        logger.info(
                            f"No improvement in validation loss for {epochs_since_last_improvement} epochs, terminating training."
                        )
                        break

        # print("lr",lr_scheduler.get_last_lr()[0])
    return best_model


def test(test_loader, model, device, res_path):
    # Load all statistics and constants
    aux_constants = utils_data.loadAllConstants(device=device)

    batch_id = 0
    for id, data in enumerate(test_loader, 0):
        # Store initial input for different models
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] predict on {id}")
        (
            input_test,
            input_surface_test,
            target_test,
            periods_test,
        ) = data
        input_test, input_surface_test, target_test = (
            input_test.to(device),
            input_surface_test.to(device),
            target_test.to(device),
        )
        model.eval()

        # Inference
        output_test = model(
            input_test,
            input_surface_test,
            aux_constants["weather_statistics"],
            aux_constants["constant_maps"],
            aux_constants["const_h"],
        )

        # Multiply w/ lsm
        if cfg.PG.TEST.USE_LSM:
            device_upper = output_test.device
            lsm_expanded = utils_data.loadLandSeaMask(
                device_upper, mask_type="sea", fill_value=0
            )

            # Multiply output_test with the land-sea mask
            output_test = output_test * lsm_expanded

        target_time = periods_test[1][batch_id]

        # Visualize
        png_path = os.path.join(res_path, "png")
        utils.mkdirs(png_path)

        # ['msl', 'u','v','t2m']
        utils.visuailze_power(
            output_test.detach().cpu().squeeze(),
            target_test.detach().cpu().squeeze(),
            step=target_time,
            path=png_path,
        )


if __name__ == "__main__":
    pass
