import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from tqdm import tqdm
from era5_data import utils, utils_data
from era5_data import score
from era5_data.config import cfg
from torch.utils import data
from datetime import datetime, timedelta
from torch import nn

# The directory of your input and output data

PATH = cfg.PG_INPUT_PATH

output_data_dir = cfg.PG_OUT_PATH

#Load pretrained model
model_24 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_24)
# model_6 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_6)
# model_3 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_3)
# model_1 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_1)

# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False

# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = cfg.GLOBAL.NUM_STREADS

# Set the behavier of cuda provider
cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

ort_session_24 = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_24, sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
# ort_session_6 = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_6, sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
# ort_session_3 = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_3, sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
# ort_session_1 = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_1, sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])


# A test for a single input frame
# desiered output: future 14 days forecast

h = cfg.PG.HORIZON
output_data_dir = os.path.join(output_data_dir, str(h))
utils.mkdirs(output_data_dir)

num = 0
# Load mean and std of the weather data
# weather_surface_mean, weather_surface_std = utils.LoadStatic()

#Prepare for the test data
test_dataset = utils_data.NetCDFDataset(nc_path=PATH,
                               data_transform=None,
                               training=False,
                               validation = False, 
                               startDate = cfg.PG.TEST.START_TIME,
                               endDate= cfg.PG.TEST.END_TIME,
                               freq=cfg.PG.TEST.FREQUENCY,
                               horizon=h)
dataset_length = len(test_dataset)

test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=cfg.PG.TEST.BATCH_SIZE,
                                          drop_last=True, shuffle=False, num_workers=0, pin_memory=True)

  # Loss function
criterion = nn.L1Loss(reduction='none')

# Dic to save rmses
rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v = dict(), dict(), dict(), dict(),dict()
rmse_surface = dict()

acc_upper_z, acc_upper_q, acc_upper_t, acc_upper_u, acc_upper_v = dict(), dict(), dict(), dict(),dict()
acc_surface = dict()


batch_id = 0
for id, data in tqdm(enumerate(test_dataloader, 0)): # 每天00:00 12:00预报h小时候的天气（single frame output）
  # Store initial input for different models
  input, input_surface, target, target_surface, periods = data

 # Required input to the pretrained model: upper ndarray(n, Z, W, H) and surface(n, W, H)
  input_24, input_surface_24 = input.numpy().astype(np.float32).squeeze(), input_surface.numpy().astype(np.float32).squeeze() # input torch.Size([1, 5, 13, 721, 1440])

  
  spaces = h // 24
  # start time
  input_time = datetime.strptime(periods[0][batch_id], '%Y%m%d%H')

  # multi-step prediction for single output
  for space in range(spaces):
    current_time = input_time + timedelta(hours = 24*(space+1))
    print("predicting on....", current_time)

    # Call the model pretrained for 24 hours forecast
    output, output_surface = ort_session_24.run(None, {'input':input_24, 'input_surface':input_surface_24})

    # Stored the output for next round forecast
    input_24, input_surface_24 = output, output_surface
    
  # make sure the predicted time step is the same as the target time step
  assert current_time == datetime.strptime(periods[1][batch_id], '%Y%m%d%H')
  target_time = periods[1][batch_id]

  output, output_surface = torch.from_numpy(output).type(torch.float32), torch.from_numpy(output_surface).type(torch.float32)

  target, target_surface = target.squeeze(), target_surface.squeeze()
  output, output_surface = output.squeeze(), output_surface.squeeze()
        #mslp, u,v,t2m 3: visualize t2m
  png_path = os.path.join(output_data_dir,"png")
  if not os.path.exists(png_path):
    os.mkdir(png_path)
  """
  utils.visuailze(output,
                    target, 
                    input.numpy().astype(np.float32).squeeze(),
                    var='t',
                    z=2,
                    step=target_time, 
                    path=png_path)

  utils.visuailze_surface(output_surface,
                          target_surface, 
                          input_surface.numpy().astype(np.float32).squeeze(),
                          var='u10',
                          step=target_time, 
                          path=png_path)

  """
  # RMSE for each variabl

  rmse_upper_z[target_time] = score.weighted_rmse_torch_channels(output[0], target[0]).numpy()
  rmse_upper_q[target_time] = score.weighted_rmse_torch_channels(output[1], target[1]).numpy()
  rmse_upper_t[target_time] = score.weighted_rmse_torch_channels(output[2], target[2]).numpy()
  rmse_upper_u[target_time] = score.weighted_rmse_torch_channels(output[3], target[3]).numpy()
  rmse_upper_v[target_time] = score.weighted_rmse_torch_channels(output[4], target[4]).numpy()
  rmse_surface[target_time] = score.weighted_rmse_torch_channels(output_surface, target_surface).numpy()

  # acc
  surface_mean, _, upper_mean, _ = utils_data.weatherStatistics_output()
  output_anomaly = output - upper_mean.squeeze(0)
  target_anomaly = target - upper_mean.squeeze(0)

  output_surface_anomaly = output_surface - surface_mean.squeeze(0)
  target_surface_anomaly = target_surface - surface_mean.squeeze(0)
  acc_upper_z[target_time] = score.weighted_acc_torch_channels(output_anomaly[0], target_anomaly[0]).numpy()
  acc_upper_q[target_time] = score.weighted_acc_torch_channels(output_anomaly[1], target_anomaly[1]).numpy()
  acc_upper_t[target_time] = score.weighted_acc_torch_channels(output_anomaly[2], target_anomaly[2]).numpy()
  acc_upper_u[target_time] = score.weighted_acc_torch_channels(output_anomaly[3], target_anomaly[3]).numpy()
  acc_upper_v[target_time] = score.weighted_acc_torch_channels(output_anomaly[4], target_anomaly[4]).numpy()

  acc_surface[target_time] = score.weighted_acc_torch_channels(output_surface_anomaly,
                                                               target_surface_anomaly).numpy()

# Save rmse,acc to csv
csv_path = os.path.join(output_data_dir, "csv")
utils.mkdirs(csv_path)

utils.save_errorScores(csv_path, rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v, rmse_surface,
                       "rmse")
utils.save_errorScores(csv_path, acc_upper_z, acc_upper_q, acc_upper_t, acc_upper_u, acc_upper_v, acc_surface, "acc")

