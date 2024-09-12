import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from era5_data import utils,utils_data
from era5_data.config import cfg
from models.pangu_model import PanguModel
import os
from torch.utils import data
from models.pangu_sample import test, train
import argparse
import time
import logging
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
"""
This is to test the performance of PEFT model
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_net', type=str, default="lora")
    parser.add_argument('--load_best', type=bool, default=True)

    args = parser.parse_args()
    starts  = time.time()
   
    PATH = cfg.PG_INPUT_PATH

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Predicting on {device}")

    output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    utils.mkdirs(output_path)

    logger_name = args.type_net + str(cfg.PG.HORIZON)
    utils.logger_info(logger_name, os.path.join(output_path, logger_name + '_test.log'))

    logger = logging.getLogger(logger_name)

    test_dataset = utils_data.NetCDFDataset(nc_path=PATH,
                                data_transform=None,
                                training=False,
                                validation = False,
                                startDate = cfg.PG.TEST.START_TIME,
                                endDate= cfg.PG.TEST.END_TIME,
                                freq=cfg.PG.TEST.FREQUENCY,
                                horizon=cfg.PG.HORIZON)

    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=cfg.PG.TEST.BATCH_SIZE,
                                            drop_last=True, shuffle=False, num_workers=0, pin_memory=False)



    torch.set_num_threads(2)

    model = PanguModel(device=device).to(device)

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
        modules_to_save=["_output_layer.conv_surface","_output_layer.conv"]
    )

    peft_model = get_peft_model(model, config)
    cpk = torch.load(os.path.join(output_path,"models/train_10.pth"))
    peft_model.load_state_dict(cpk['model'])#, strict=False)#map_location='cuda:0'

    logger.info("Begin Test")
    msg = '\n'
    msg += utils.torch_summarize(peft_model, show_weights=False)
    logger.info(msg)

    output_path = os.path.join(output_path, "test")
    utils.mkdirs(output_path)
    test(test_loader=test_dataloader,
             model = peft_model,
             device=peft_model.device,
             res_path = output_path)
