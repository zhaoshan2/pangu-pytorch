import sys
from .ordered_easydict import OrderedEasyDict as edict
import numpy as np
import os
import torch

__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.BATCH_SZIE = 1
for dirs in ['/home/scc/om1434/pangu_zhaoshan2', 'your_path']:
    if os.path.exists(dirs):
        __C.GLOBAL.PATH = dirs
assert __C.GLOBAL.PATH is not None
__C.GLOBAL.SEED =99
__C.GLOBAL.NUM_STREADS = 16


# __C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
__C.PG_INPUT_PATH = '/home/scc/om1434/pangu_zhaoshan2/data_storage_home/data/pangu'
assert __C.PG_INPUT_PATH is not None

__C.PG_OUT_PATH = os.path.join(__C.GLOBAL.PATH,'result')
assert __C.PG_OUT_PATH is not None

__C.ERA5_UPPER_LEVELS = ['1000','925','850', '700','600','500','400', '300','250', '200','150','100', '50']
__C.ERA5_SURFACE_VARIABLES = ['msl','u10','v10','t2m']
__C.ERA5_UPPER_VARIABLES = ['z','q','t','u','v']


__C.PG = edict()

__C.PG.TRAIN = edict()

__C.PG.HORIZON = 24

__C.PG.TRAIN.EPOCHS = 100
__C.PG.TRAIN.LR = 5e-6 #5e-4
__C.PG.TRAIN.WEIGHT_DECAY = 3e-6
__C.PG.TRAIN.START_TIME = '20150101'
__C.PG.TRAIN.END_TIME = '20171231' #'20171231'
__C.PG.TRAIN.FREQUENCY = '12H'
__C.PG.TRAIN.BATCH_SIZE = 2
__C.PG.TRAIN.UPPER_WEIGHTS = [3.00, 0.60, 1.50, 0.77, 0.54]
__C.PG.TRAIN.SURFACE_WEIGHTS = [1.50, 0.77, 0.66, 3.00]
__C.PG.TRAIN.SAVE_INTERVAL = 1
__C.PG.VAL = edict()


__C.PG.VAL.START_TIME = '20190101'
__C.PG.VAL.END_TIME = '20191231'
__C.PG.VAL.FREQUENCY = '12H'
__C.PG.VAL.BATCH_SIZE = 1
__C.PG.VAL.INTERVAL = 1


__C.PG.TEST = edict()

__C.PG.TEST.START_TIME = '20190701'
__C.PG.TEST.END_TIME = '20190702'
__C.PG.TEST.FREQUENCY = '12H'
__C.PG.TEST.BATCH_SIZE = 1

__C.PG.BENCHMARK = edict()

__C.PG.BENCHMARK.PRETRAIN_24 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_24.onnx')
__C.PG.BENCHMARK.PRETRAIN_6 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_6.onnx')
__C.PG.BENCHMARK.PRETRAIN_3 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_3.onnx')
__C.PG.BENCHMARK.PRETRAIN_1 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_1.onnx')

__C.PG.BENCHMARK.PRETRAIN_24_fp16 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model_fp16/pangu_weather_24_fp16.onnx')

__C.PG.BENCHMARK.PRETRAIN_24_torch = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_24_torch.pth')
  
   
__C.MODEL = edict()


