import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from era5_data.config import cfg

from typing import Tuple
import random
from torch.utils import data


class PowerDataset(data.Dataset):
    """Dataset class containing both weather data (input) and power data (target)"""

    def __init__(
        self,
        era5_path=cfg.PG_INPUT_PATH,
        power_path=cfg.PG_POWER_PATH,
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
        self.era5_path = era5_path

        # Prepare the datetime objects for training, validation, and test
        self.training = training
        self.validation = validation
        self.data_transform = data_transform

        self.keys = list(pd.date_range(start=startDate, end=endDate, freq=freq))
        self.length = len(self.keys) - horizon // 12 - 1

        self.input_upper_dataset, self.input_surface_dataset = self._get_era5_data()
        self.power_dataset = self._get_power_data()

        random.seed(seed)

    def nctonumpy(self, dataset_upper, dataset_surface):
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

    def _get_era5_data(self) -> Tuple[xr.Dataset, xr.Dataset]:
        era5_data = xr.open_dataset(cfg.ERA5_PATH, engine="zarr")
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
        era5_data = era5_data[list(variable_mapping.keys())].rename(variable_mapping)

        input_upper_dataset = era5_data[upper_variables]
        input_surface_dataset = era5_data[surface_variables]

        return input_upper_dataset, input_surface_dataset

    def _get_power_data(self) -> None:
        """
        Load offshore power data based on the specified location type.

        Parameters:
        location_type (str): The type of location for which to load data.
                            Should be 'onshore' or 'offshore'.

        Returns:
        xr.Dataset: The loaded power dataset.
        """

        # TODO(EliasKng): Continue here

        return None

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
