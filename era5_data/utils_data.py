import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from era5_data.config import cfg

from typing import Tuple
import torch
import random
from torch.utils import data
from typing import Optional


class DataPrefetcher:
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.length = len(self.loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.__preload__()

    def __preload__(self):
        try:
            (
                self.input,
                self.input_surface,
                self.target,
                self.target_surface,
                self.periods,
            ) = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            (
                self.input,
                self.input_surface,
                self.target,
                self.target_surface,
                self.periods,
            ) = next(self.dataiter)

        with torch.cuda.stream(self.stream):
            self.target = self.target.cuda(non_blocking=True)
            self.target_surface = self.target_surface.cuda(non_blocking=True)
            self.input = self.input.cuda(non_blocking=True)
            self.input_surface = self.input_surface.cuda(non_blocking=True)
            self.periods = self.periods.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        self.__preload__()
        return (
            self.input,
            self.input_surface,
            self.target,
            self.target_surface,
            self.periods,
        )

    def __len__(self):
        """Return the number of images."""
        return self.length


class NetCDFDataset(data.Dataset):
    """Dataset class for the era5 upper and surface variables."""

    def __init__(
        self,
        nc_path=cfg.PG_INPUT_PATH,
        data_transform=None,
        seed=1234,
        training=True,
        validation=False,
        startDate="20150101",
        endDate="20150102",
        freq="h",
        horizon=5,
    ):
        """Initialize."""
        self.horizon = horizon
        self.nc_path = nc_path
        """
        To do
        if start and end is valid date, if the date can be found in the downloaded files, length >= 0

        """
        # Prepare the datetime objects for training, validation, and test
        self.training = training
        self.validation = validation
        self.data_transform = data_transform

        if training:
            self.keys = list(pd.date_range(start=startDate, end=endDate, freq=freq))
            # self.keys = (list(set(self.keys))) #disordered keys
            # total length that we can predict
            """
            To do
            length should >=0 horizon <= len
            """
        elif validation:
            self.keys = list(pd.date_range(start=startDate, end=endDate, freq=freq))
            # self.keys = (list(set(self.keys)))

        else:
            self.keys = list(pd.date_range(start=startDate, end=endDate, freq=freq))
            # self.keys = (list(set(self.keys)))
            # end_time = self.keys[0] + timedelta(hours = self.horizon)
        self.length = len(self.keys) - horizon // 12 - 1

        self.input_upper_dataset, self.input_surface_dataset = self._get_zarr_data()

        random.seed(seed)

    @staticmethod
    def nctonumpy(dataset_upper, dataset_surface):
        """
        Input
            xr.Dataset upper, surface
        Return
            numpy array upper, surface
        """

        upper_z = dataset_upper["z"].values.astype(np.float32)  # (13,721,1440)
        upper_q = dataset_upper["q"].values.astype(np.float32)
        upper_t = dataset_upper["t"].values.astype(np.float32)
        upper_u = dataset_upper["u"].values.astype(np.float32)
        upper_v = dataset_upper["v"].values.astype(np.float32)
        upper = np.concatenate(
            (
                upper_z[np.newaxis, ...],
                upper_q[np.newaxis, ...],
                upper_t[np.newaxis, ...],
                upper_u[np.newaxis, ...],
                upper_v[np.newaxis, ...],
            ),
            axis=0,
        )
        assert upper.shape == (5, 13, 721, 1440)
        # levels in descending order, require new memery space
        upper = upper[:, ::-1, :, :].copy()

        surface_mslp = dataset_surface["msl"].values.astype(np.float32)  # (721,1440)
        surface_u10 = dataset_surface["u10"].values.astype(np.float32)
        surface_v10 = dataset_surface["v10"].values.astype(np.float32)
        surface_t2m = dataset_surface["t2m"].values.astype(np.float32)
        surface = np.concatenate(
            (
                surface_mslp[np.newaxis, ...],
                surface_u10[np.newaxis, ...],
                surface_v10[np.newaxis, ...],
                surface_t2m[np.newaxis, ...],
            ),
            axis=0,
        )
        assert surface.shape == (4, 721, 1440)

        return upper, surface

    @staticmethod
    def _get_zarr_data() -> Tuple[xr.Dataset, xr.Dataset]:
        zarr_data = xr.open_dataset(cfg.ERA5_PATH, engine="zarr")
        variable_mapping = {
            "geopotential": "z",
            "specific_humidity": "q",
            "temperature": "t",
            "u_component_of_wind": "u",
            "v_component_of_wind": "v",
            "mean_sea_level_pressure": "msl",
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
            "2m_temperature": "t2m",
        }

        surface_variables = ["msl", "u10", "v10", "t2m"]
        upper_variables = ["z", "q", "t", "u", "v"]

        # Select and rename variables according to the mapping
        zarr_data = zarr_data[list(variable_mapping.keys())].rename(variable_mapping)

        input_upper_dataset = zarr_data[upper_variables]
        input_surface_dataset = zarr_data[surface_variables]

        return input_upper_dataset, input_surface_dataset

    def LoadData(
        self, key
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[str, str]]:
        """
        Input
            key: datetime object, input time
        Return
            input: numpy
            input_surface: numpy
            target: numpy label
            target_surface: numpy label
            (start_time_str, end_time_str): string, datetime(target time - input time) = horizon
        """
        # start_time datetime obj
        start_time = key
        # convert datetime obj to string for matching file name and return key
        start_time_str = datetime.strftime(key, "%Y%m%d%H")

        # target time = start time + horizon
        end_time = key + timedelta(hours=self.horizon)
        end_time_str = end_time.strftime("%Y%m%d%H")

        # Get datasets
        input_surface_dataset = self.input_surface_dataset.sel(time=start_time)
        input_upper_dataset = self.input_upper_dataset.sel(time=start_time)
        target_surface_dataset = self.input_surface_dataset.sel(time=end_time)
        target_upper_dataset = self.input_upper_dataset.sel(time=end_time)

        # make sure upper and surface variables are at the same time
        assert input_surface_dataset["time"] == input_upper_dataset["time"]
        assert target_upper_dataset["time"] == target_surface_dataset["time"]

        # datasets to target numpy
        input, input_surface = self.nctonumpy(
            input_upper_dataset, input_surface_dataset
        )
        target, target_surface = self.nctonumpy(
            target_upper_dataset, target_surface_dataset
        )

        return (
            input,
            input_surface,
            target,
            target_surface,
            (start_time_str, end_time_str),
        )

    def __getitem__(self, index):
        """Return input frames, target frames, and its corresponding time steps."""
        if self.training:
            iii = self.keys[index]
            input, input_surface, target, target_surface, periods = self.LoadData(iii)

            if self.data_transform is not None:
                input = self.data_transform(input)
                input_surface = self.data_transform(input_surface)

        else:
            iii = self.keys[index]
            input, input_surface, target, target_surface, periods = self.LoadData(iii)

        return input, input_surface, target, target_surface, periods

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


def weatherStatistics_output(
    filepath=os.path.join(cfg.PG_INPUT_PATH, "aux_data"), device="cpu"
):
    """
    :return:1, 5, 13, 1, 1
    """
    surface_mean = np.load(os.path.join(filepath, "surface_mean.npy")).astype(
        np.float32
    )
    surface_std = np.load(os.path.join(filepath, "surface_std.npy")).astype(np.float32)
    surface_mean = torch.from_numpy(surface_mean)
    surface_std = torch.from_numpy(surface_std)
    surface_mean = surface_mean.view(1, 4, 1, 1)
    surface_std = surface_std.view(1, 4, 1, 1)

    upper_mean = np.load(os.path.join(filepath, "upper_mean.npy")).astype(
        np.float32
    )  # (13,1,1,5)
    upper_mean = upper_mean[::-1, :, :, :].copy()
    upper_mean = np.transpose(upper_mean, (1, 3, 0, 2))  # (1,5,13, 1)
    upper_mean = torch.from_numpy(upper_mean)

    upper_std = np.load(os.path.join(filepath, "upper_std.npy")).astype(np.float32)
    upper_std = upper_std[::-1, :, :, :].copy()
    upper_std = np.transpose(upper_std, (1, 3, 0, 2))
    upper_std = torch.from_numpy(upper_std)

    return (
        surface_mean.to(device),
        surface_std.to(device),
        upper_mean[..., None].to(device),
        upper_std[..., None].to(device),
    )


def weatherStatistics_input(
    filepath=os.path.join(cfg.PG_INPUT_PATH, "aux_data"), device="cpu"
):
    """
    :return:13, 1, 1, 5
    """
    surface_mean = np.load(os.path.join(filepath, "surface_mean.npy")).astype(
        np.float32
    )
    surface_std = np.load(os.path.join(filepath, "surface_std.npy")).astype(np.float32)
    surface_mean = torch.from_numpy(surface_mean)
    surface_std = torch.from_numpy(surface_std)

    upper_mean = np.load(os.path.join(filepath, "upper_mean.npy")).astype(np.float32)
    upper_std = np.load(os.path.join(filepath, "upper_std.npy")).astype(np.float32)
    upper_mean = torch.from_numpy(upper_mean)
    upper_std = torch.from_numpy(upper_std)

    return (
        surface_mean.to(device),
        surface_std.to(device),
        upper_mean.to(device),
        upper_std.to(device),
    )


def LoadConstantMask(filepath="/home/code/Pangu-Weather/constant_masks", device="cpu"):
    land_mask = np.load(os.path.join(filepath, "land_mask.npy")).astype(np.float32)
    soil_type = np.load(os.path.join(filepath, "soil_type.npy")).astype(np.float32)
    topography = np.load(os.path.join(filepath, "topography.npy")).astype(np.float32)
    land_mask = torch.from_numpy(land_mask)  # ([721, 1440])
    soil_type = torch.from_numpy(soil_type)  # ([721, 1440])
    topography = torch.from_numpy(topography)  # ([721, 1440])

    return (
        land_mask[None, None, ...].to(device),
        soil_type[None, None, ...].to(device),
        topography[None, None, ...].to(device),
    )  # torch.Size([1, 1, 721, 1440])


def LoadConstantMask3(
    filepath=os.path.join(cfg.PG_INPUT_PATH, "aux_data"), device="cpu"
):
    mask = np.load(os.path.join(filepath, "constantMaks3.npy")).astype(np.float32)
    mask = torch.from_numpy(mask)
    return mask.to(device)


def computeStatistics(train_loader):
    # prepare for the statistics
    weather_surface_mean, weather_surface_std = (
        torch.zeros(1, 4, 1, 1),
        torch.zeros(1, 4, 1, 1),
    )
    weather_mean, weather_std = torch.zeros(1, 5, 13, 1, 1), torch.zeros(1, 5, 13, 1, 1)
    for id, train_data in enumerate(train_loader, 0):
        input, input_surface, _, _, _ = train_data
        weather_surface_mean += torch.mean(input_surface, dim=(-1, -2), keepdim=True)
        weather_surface_std += torch.std(input_surface, dim=(-1, -2), keepdim=True)
        weather_mean += torch.mean(input, dim=(-1, -2), keepdim=True)
        weather_std += torch.std(input, dim=(-1, -2), keepdim=True)  # (1,5,13,)
    weather_surface_mean, weather_surface_std, weather_mean, weather_std = (
        weather_surface_mean / len(train_loader),
        weather_surface_std / len(train_loader),
        weather_mean / len(train_loader),
        weather_std / len(train_loader),
    )

    return weather_surface_mean, weather_surface_std, weather_mean, weather_std


def loadConstMask_h(filepath=os.path.join(cfg.PG_INPUT_PATH, "aux_data"), device="cpu"):
    mask_h = np.load(os.path.join(filepath, "Constant_17_output_0.npy")).astype(
        np.float32
    )
    mask_h = torch.from_numpy(mask_h)
    return mask_h.to(device)


def loadVariableWeights(device="cpu"):
    upper_weights = (
        torch.FloatTensor(cfg.PG.TRAIN.UPPER_WEIGHTS)
        .unsqueeze(0)
        .unsqueeze(2)
        .unsqueeze(3)
        .unsqueeze(4)
    )
    surface_weights = (
        torch.FloatTensor(cfg.PG.TRAIN.SURFACE_WEIGHTS)
        .unsqueeze(0)
        .unsqueeze(2)
        .unsqueeze(3)
    )
    return upper_weights.to(device), surface_weights.to(device)


def loadAllConstants(device):
    constants = dict()
    constants["weather_statistics"] = weatherStatistics_input(
        device=device
    )  # height has inversed shape, order is reversed in model
    constants["weather_statistics_last"] = weatherStatistics_output(device=device)
    # constants['constant_maps'] = LoadConstantMask(device=device)
    constants["constant_maps"] = LoadConstantMask3(
        device=device
    )  # not able to be equal
    constants["variable_weights"] = loadVariableWeights(device=device)
    constants["const_h"] = loadConstMask_h(device=device)

    return constants


def loadLandSeaMasksPangu(
    device_upper: torch.device,
    device_surface: torch.device,
    mask_type: str = "sea",
    fill_value=float("nan"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the land-sea mask (LSM) for Europe (as used in CDS data) and expand it to the required shape.

    Parameters
    ----------
    device_upper : torch.device
        The device to load the upper-level land-sea mask onto.
    device_surface : torch.device
        The device to load the surface-level land-sea mask onto.
    mask_type : str, optional
        The type of mask to return ('land' or 'sea'). Default is 'sea'.
        If e.g., 'sea', then the mask will be 1 for sea (Europe) and  NaN for everything else (not europe, land).
    fill_value : int, optional
        The value to fill the non 1s with. Default is NaN.
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The expanded land-sea masks for upper and surface levels.
    """
    # Load the land-sea mask (LSM) from the dataset
    lsm: xr.DataArray = xr.open_dataset(cfg.LSM_PATH, engine="zarr").lsm  # [721, 1440]
    if mask_type == "land":
        lsm = xr.where(lsm.isnull(), fill_value, xr.where(lsm == 1, 1, fill_value))
    elif mask_type == "sea":
        lsm = xr.where(lsm.isnull(), fill_value, xr.where(lsm == 0, 1, fill_value))
    else:
        raise ValueError("mask_type must be either 'land' or 'sea'")

    # Convert the LSM to a NumPy array and flip it upside down (since xr exports it upside down)
    lsm_np: np.ndarray = lsm.values
    lsm_np = np.flipud(lsm_np)  # [721, 1440]

    # Expand lsm_np to [1, 5, 13, 721, 1440] by copying the original array
    lsm_expanded_np = np.expand_dims(lsm_np, axis=(0, 1, 2))  # [1, 1, 1, 721, 1440]
    lsm_expanded_np = np.tile(
        lsm_expanded_np, (1, 5, 13, 1, 1)
    )  # [1, 5, 13, 721, 1440]
    lsm_expanded = torch.tensor(lsm_expanded_np, device=device_upper)

    # Expand lsm_np to [1, 4, 721, 1440] by copying the original array
    lsm_surface_expanded_np = np.expand_dims(lsm_np, axis=(0, 1))  # [1, 1, 721, 1440]
    lsm_surface_expanded_np = np.tile(
        lsm_surface_expanded_np, (1, 4, 1, 1)
    )  # [1, 4, 721, 1440]
    lsm_surface_expanded = torch.tensor(lsm_surface_expanded_np, device=device_surface)

    return lsm_expanded, lsm_surface_expanded


def loadLandSeaMask(
    device: Optional[torch.device] = None,
    mask_type: str = "sea",
    fill_value: float = float("nan"),
) -> torch.Tensor:
    # Load the land-sea mask (LSM) from the dataset
    lsm: xr.DataArray = xr.open_dataset(cfg.LSM_PATH, engine="zarr").lsm  # [721, 1440]
    if mask_type == "land":
        lsm = xr.where(lsm.isnull(), fill_value, xr.where(lsm == 1, 1, fill_value))
    elif mask_type == "sea":
        lsm = xr.where(lsm.isnull(), fill_value, xr.where(lsm == 0, 1, fill_value))
    else:
        raise ValueError("mask_type must be either 'land' or 'sea'")

    # Convert the LSM to a NumPy array and flip it upside down (since xr exports it upside down)
    lsm_np: np.ndarray = lsm.values
    lsm_np = np.flipud(lsm_np).copy()  # [721, 1440]

    lsm_expanded_np = np.expand_dims(lsm_np, axis=(0))  # [1, 721, 1440]

    return torch.tensor(lsm_expanded_np, device=device)


def normData(upper, surface, statistics):
    surface_mean, surface_std, upper_mean, upper_std = (
        statistics[0],
        statistics[1],
        statistics[2],
        statistics[3],
    )

    upper = (upper - upper_mean) / upper_std
    surface = (surface - surface_mean) / surface_std
    return upper, surface


def normBackData(upper, surface, statistics):
    surface_mean, surface_std, upper_mean, upper_std = (
        statistics[0],
        statistics[1],
        statistics[2],
        statistics[3],
    )
    upper = upper * upper_std + upper_mean
    surface = surface * surface_std + surface_mean

    return upper, surface


def normBackDataSurface(surface, statistics):
    surface_mean, surface_std = (
        statistics[0],
        statistics[1],
    )
    surface = surface * surface_std + surface_mean

    return surface


if __name__ == "__main__":
    # dataset_path = cfg.PG_INPUT_PATH
    # means, std = LoadStatic(os.path.join(dataset_path, 'aux_data'))
    # print(means.shape) #(1, 21, 1, 1)
    a, b, c, d = weatherStatistics_input()
    print(a.shape)
