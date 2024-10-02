import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from era5_data import utils, utils_data
from era5_data.config import cfg
from models.pangu_model import PanguModel
import os
import torch
import os
from torch.utils import data
from models.pangu_sample import test
import argparse
import time
import logging

if __name__ == "__main__":
    """
    check the re-implemented model performance
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_net", type=str, default="reproduce_mask0")
    args = parser.parse_args()
    starts = time.time()

    PATH = cfg.PG_INPUT_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Predicting on {device}")

    output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    utils.mkdirs(output_path)

    logger_name = args.type_net + str(cfg.PG.HORIZON)
    utils.logger_info(logger_name, os.path.join(output_path, logger_name + "_test.log"))

    logger = logging.getLogger(logger_name)

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

    torch.set_num_threads(16)

    model = PanguModel(device=device).to(device)

    checkpoint = torch.load(
        cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])

    logger.info("Begin Test")
    msg = "\n"
    # msg += utils.torch_summarize(model, show_weights=False)
    logger.info(msg)
    output_path = os.path.join(output_path, "test")
    utils.mkdirs(output_path)

    test(
        test_loader=test_dataloader,
        model=model,
        device=model.device,
        res_path=output_path,
    )
