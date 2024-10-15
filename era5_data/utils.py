import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from era5_data.config import cfg
from era5_data import utils_data as utils_data
import torch

from torch.nn.modules.module import _addindent
import matplotlib.pyplot as plt
import logging


def logger_info(logger_name, log_path="default_logger.log"):
    """set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    """
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print("LogHandlers exist!")
    else:
        print("LogHandlers setup!")
        level = logging.INFO
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d : %(message)s", datefmt="%y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


"""
# --------------------------------------------
# print to file and std_out simultaneously
# --------------------------------------------
"""


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass


def visuailze(output, target, input, var, z, step, path):
    # levels = np.linspace(-30, 90, 9)
    variables = cfg.ERA5_UPPER_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(16, 2))

    max_bias = _calc_max_bias(output[var, z, :, :], target[var, z, :, :])

    ax_1 = fig.add_subplot(141)
    plot3 = ax_1.imshow(input[var, z, :, :], cmap="RdBu")
    plt.colorbar(plot3, ax=ax_1, fraction=0.05, pad=0.05)
    ax_1.title.set_text("input")

    ax_2 = fig.add_subplot(142)
    plot2 = ax_2.imshow(target[var, z, :, :], cmap="RdBu")
    plt.colorbar(plot2, ax=ax_2, fraction=0.05, pad=0.05)
    ax_2.title.set_text("gt")

    ax_3 = fig.add_subplot(143)
    plot1 = ax_3.imshow(
        output[var, z, :, :], cmap="RdBu"
    )  # , levels = levels, extend = 'min')
    plt.colorbar(plot1, ax=ax_3, fraction=0.05, pad=0.05)
    ax_3.title.set_text("pred")

    ax_4 = fig.add_subplot(144)
    plot4 = ax_4.imshow(
        output[var, z, :, :] - target[var, z, :, :],
        cmap="RdBu",
        vmin=-max_bias,
        vmax=max_bias,
    )
    plt.colorbar(plot4, ax=ax_4, fraction=0.05, pad=0.05)
    ax_4.title.set_text("bias")

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, "{}_{}_Z{}".format(step, variables[var], z)))


def visuailze_surface(output, target, input, var, step, path):
    variables = cfg.ERA5_SURFACE_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(16, 2))

    max_bias = _calc_max_bias(output[var, :, :], target[var, :, :])

    ax_1 = fig.add_subplot(141)
    plot3 = ax_1.imshow(input[var, :, :], cmap="RdBu")
    plt.colorbar(plot3, ax=ax_1, fraction=0.05, pad=0.05)
    ax_1.title.set_text("input")

    ax_2 = fig.add_subplot(142)
    plot2 = ax_2.imshow(target[var, :, :], cmap="RdBu")
    plt.colorbar(plot2, ax=ax_2, fraction=0.05, pad=0.05)
    ax_2.title.set_text("gt")

    ax_3 = fig.add_subplot(143)
    plot1 = ax_3.imshow(
        output[var, :, :], cmap="RdBu"
    )  # , levels = levels, extend = 'min')
    plt.colorbar(plot1, ax=ax_3, fraction=0.05, pad=0.05)
    ax_3.title.set_text("pred")

    ax_4 = fig.add_subplot(144)
    plot4 = ax_4.imshow(
        output[var, :, :] - target[var, :, :],
        cmap="RdBu",
        vmin=-max_bias,
        vmax=max_bias,
    )
    plt.colorbar(plot4, ax=ax_4, fraction=0.05, pad=0.05)
    ax_4.title.set_text("bias")

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, "{}_{}".format(step, variables[var])))
    plt.close()


def visualize_windspeed(output, target, input, step, path):
    variables = cfg.ERA5_SURFACE_VARIABLES
    var1 = variables.index("u10")
    var2 = variables.index("v10")

    wind_speed_input = torch.sqrt(input[var1, :, :] ** 2 + input[var2, :, :] ** 2)
    wind_speed_input = prepare_europe(wind_speed_input)

    wind_speed_output = torch.sqrt(output[var1, :, :] ** 2 + output[var2, :, :] ** 2)
    wind_speed_output = prepare_europe(wind_speed_output)

    wind_speed_target = torch.sqrt(target[var1, :, :] ** 2 + target[var2, :, :] ** 2)
    wind_speed_target = prepare_europe(wind_speed_target)

    max_bias = _calc_max_bias(wind_speed_output, wind_speed_target)

    fig = plt.figure(figsize=(12, 2))

    ax_1 = fig.add_subplot(141)
    plot3 = ax_1.imshow(wind_speed_input, cmap="RdBu")
    plt.colorbar(plot3, ax=ax_1, fraction=0.05, pad=0.05)
    ax_1.title.set_text("input")

    ax_2 = fig.add_subplot(142)
    plot2 = ax_2.imshow(wind_speed_target, cmap="RdBu")
    plt.colorbar(plot2, ax=ax_2, fraction=0.05, pad=0.05)
    ax_2.title.set_text("gt (Δ 24h)")

    ax_3 = fig.add_subplot(143)
    plot1 = ax_3.imshow(wind_speed_output, cmap="RdBu")
    plt.colorbar(plot1, ax=ax_3, fraction=0.05, pad=0.05)
    ax_3.title.set_text("pred (Δ 24h)")

    ax_4 = fig.add_subplot(144)
    plot4 = ax_4.imshow(
        wind_speed_output - wind_speed_target,
        cmap="RdBu",
        vmin=-max_bias,
        vmax=max_bias,
    )
    plt.colorbar(plot4, ax=ax_4, fraction=0.05, pad=0.05)
    ax_4.title.set_text("bias")

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, "{}_{}".format(step, "wind_speed")))
    plt.close()


def visuailze_power(output, target, input, step, path):
    variables = cfg.ERA5_SURFACE_VARIABLES
    var1 = variables.index("u10")
    var2 = variables.index("v10")
    wind_speed = torch.sqrt(input[var1, :, :] ** 2 + input[var2, :, :] ** 2)

    wind_speed = prepare_europe(wind_speed)
    output = prepare_europe(output)
    target = prepare_europe(target)

    max_bias = _calc_max_bias(output, target)

    fig = plt.figure(figsize=(12, 2))

    ax_1 = fig.add_subplot(141)
    plot3 = ax_1.imshow(wind_speed, cmap="RdBu")
    plt.colorbar(plot3, ax=ax_1, fraction=0.05, pad=0.05)
    ax_1.title.set_text("input[wind speed]")

    ax_2 = fig.add_subplot(142)
    plot2 = ax_2.imshow(target, cmap="RdBu")
    plt.colorbar(plot2, ax=ax_2, fraction=0.05, pad=0.05)
    ax_2.title.set_text("gt")

    ax_3 = fig.add_subplot(143)
    plot1 = ax_3.imshow(output, cmap="RdBu")  # , levels = levels, extend = 'min')
    plt.colorbar(plot1, ax=ax_3, fraction=0.05, pad=0.05)
    ax_3.title.set_text("pred")

    ax_4 = fig.add_subplot(144)
    plot4 = ax_4.imshow(output - target, cmap="RdBu", vmin=-max_bias, vmax=max_bias)
    plt.colorbar(plot4, ax=ax_4, fraction=0.05, pad=0.05)
    ax_4.title.set_text("bias")

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, "{}_power".format(step)))
    plt.close()


def visuailze_all(
    output_power,
    target_power,
    input,
    output_pangu_surface,
    target_pangu_surface,
    step,
    path,
):
    """Visualizes both wind_speeds (pangu) and power predictions."""
    variables = cfg.ERA5_SURFACE_VARIABLES
    var1 = variables.index("u10")
    var2 = variables.index("v10")

    input_ws = _calc_wind_speed(input[var1, :, :], input[var2, :, :])
    target_ws = _calc_wind_speed(
        target_pangu_surface[var1, :, :], target_pangu_surface[var2, :, :]
    )
    output_ws = _calc_wind_speed(
        output_pangu_surface[var1, :, :], output_pangu_surface[var2, :, :]
    )

    input_ws = prepare_europe(input_ws)
    target_ws = prepare_europe(target_ws)
    output_ws = prepare_europe(output_ws)
    target_power = prepare_europe(target_power)
    output_power = prepare_europe(output_power)

    max_bias_ws = _calc_max_bias(output_ws, target_ws)
    max_bias_power = _calc_max_bias(output_power, target_power)

    fig = plt.figure(figsize=(12, 4))

    ax_1 = fig.add_subplot(241)
    plot_1 = ax_1.imshow(input_ws, cmap="RdBu")
    plt.colorbar(plot_1, ax=ax_1, fraction=0.05, pad=0.05)
    ax_1.title.set_text("input[wind speed]")

    ax_2 = fig.add_subplot(242)
    plot_2 = ax_2.imshow(target_ws, cmap="RdBu")
    plt.colorbar(plot_2, ax=ax_2, fraction=0.05, pad=0.05)
    ax_2.title.set_text("gt wind speed")

    ax_3 = fig.add_subplot(243)
    plot_3 = ax_3.imshow(output_ws, cmap="RdBu")
    plt.colorbar(plot_3, ax=ax_3, fraction=0.05, pad=0.05)
    ax_3.title.set_text("pred wind speed")

    ax_4 = fig.add_subplot(244)
    plot_4 = ax_4.imshow(
        output_ws - target_ws, cmap="RdBu", vmin=-max_bias_ws, vmax=max_bias_ws
    )
    plt.colorbar(plot_4, ax=ax_4, fraction=0.05, pad=0.05)
    ax_4.title.set_text("bias wind speed")

    ax_6 = fig.add_subplot(246)
    plot_6 = ax_6.imshow(target_power, cmap="RdBu")
    plt.colorbar(plot_6, ax=ax_6, fraction=0.05, pad=0.05)
    ax_6.title.set_text("gt power")

    ax_7 = fig.add_subplot(247)
    plot_7 = ax_7.imshow(
        output_power, cmap="RdBu"
    )  # , levels = levels, extend = 'min')
    plt.colorbar(plot_7, ax=ax_7, fraction=0.05, pad=0.05)
    ax_7.title.set_text("pred power")

    ax_8 = fig.add_subplot(248)
    plot_8 = ax_8.imshow(
        output_power - target_power,
        cmap="RdBu",
        vmin=-max_bias_power,
        vmax=max_bias_power,
    )
    plt.colorbar(plot_8, ax=ax_8, fraction=0.05, pad=0.05)
    ax_8.title.set_text("bias power")

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, "{}_power".format(step)))
    plt.close()


def _calc_max_bias(output, target):
    """Calculate the maximum bias between the output and target. Used for bias color scale"""
    bias = output - target
    bias_masked = bias[~torch.isnan(bias)]
    max_bias = torch.max(torch.abs(bias_masked)).item()
    return max_bias


def _calc_wind_speed(u, v):
    return torch.sqrt(u**2 + v**2)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def torch_summarize(
    model, show_weights=False, show_parameters=False, show_gradients=False
):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + " (\n"
    total_params = sum(
        [
            np.prod(p.size())
            for p in filter(lambda p: p.requires_grad, model.parameters())
        ]
    )
    tmpstr += ", total parameters={}".format(total_params)
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential,
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, module.parameters())
            ]
        )
        weights = tuple(
            [
                tuple(p.size())
                for p in filter(lambda p: p.requires_grad, module.parameters())
            ]
        )
        grads = tuple(
            [
                str(p.requires_grad)
                for p in filter(lambda p: p.requires_grad, module.parameters())
            ]
        )

        tmpstr += "  (" + key + "): " + modstr
        if show_weights:
            tmpstr += ", weights={}".format(weights)
        if show_parameters:
            tmpstr += ", parameters={}".format(params)
        if show_gradients:
            tmpstr += ", gradients={}".format(grads)
        tmpstr += "\n"

    tmpstr = tmpstr + ")"
    return tmpstr


def save_errorScores(csv_path, z, q, t, u, v, surface, error):
    score_upper_z = pd.DataFrame.from_dict(
        z, orient="index", columns=cfg.ERA5_UPPER_LEVELS
    )
    score_upper_q = pd.DataFrame.from_dict(
        q, orient="index", columns=cfg.ERA5_UPPER_LEVELS
    )
    score_upper_t = pd.DataFrame.from_dict(
        t, orient="index", columns=cfg.ERA5_UPPER_LEVELS
    )
    score_upper_u = pd.DataFrame.from_dict(
        u, orient="index", columns=cfg.ERA5_UPPER_LEVELS
    )
    score_upper_v = pd.DataFrame.from_dict(
        v, orient="index", columns=cfg.ERA5_UPPER_LEVELS
    )
    score_surface = pd.DataFrame.from_dict(
        surface, orient="index", columns=cfg.ERA5_SURFACE_VARIABLES
    )

    score_upper_z.to_csv("{}/{}.csv".format(csv_path, f"{error}_upper_z"))
    score_upper_q.to_csv("{}/{}.csv".format(csv_path, f"{error}_upper_q"))
    score_upper_t.to_csv("{}/{}.csv".format(csv_path, f"{error}_upper_t"))
    score_upper_u.to_csv("{}/{}.csv".format(csv_path, f"{error}_upper_u"))
    score_upper_v.to_csv("{}/{}.csv".format(csv_path, f"{error}_upper_v"))
    score_surface.to_csv("{}/{}.csv".format(csv_path, f"{error}_surface"))


def prepare_europe(data: torch.Tensor) -> torch.Tensor:
    """Cut out Europe area from the data and replace land area with NaN."""
    lsm = utils_data.loadLandSeaMask(
        device=None, mask_type="sea", fill_value=float("nan")
    )
    # Cut out Europe area
    data = data * lsm
    data = data.squeeze()
    data = torch.roll(data, shifts=88, dims=1)
    data = torch.roll(data, shifts=-70, dims=0)
    data = data[0:185, 0:271]
    return data


if __name__ == "__main__":
    """

    s_transforms = []

    s_transforms.append(T.RandomHorizontalFlip())

    s_transforms.append(T.RandomVerticalFlip())
    s_transforms = T.Compose(s_transforms)
    s_transforms = None

    nc_dataset = NetCDFDataset(dataset_path,
                               data_transform=None,
                               training=False,
                               validation = True,
                               startDate = '20150101',
                               endDate='20150102',
                               freq='H',
                               horizon=5)
    nc_dataloader = data.DataLoader(dataset=nc_dataset, batch_size=2,
                                          drop_last=True, shuffle=True, num_workers=0, pin_memory=True)

    print('Total length is', len(nc_dataset))


    start_time = time.time()
    nc_dataloader = iter(nc_dataloader)
    for i in range(2):
        input, input_surface, target, target_surface, periods = next(nc_dataloader)
        print(input.shape) #torch.Size([1, 5, 13, 721, 1440])
        print(input_surface.shape) #torch.Size([1, 4, 721, 1440])
        print(target.shape) #torch.Size([1, 5, 13, 721, 1440])
        print(target_surface.shape) #torch.Size([1, 4, 721, 1440])


        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        plot1 = ax1.contourf(input[0,0,0])
        plt.colorbar(plot1,ax=ax1)
        ax1.title.set_text('input 0')

        ax2 = fig.add_subplot(222)
        plot2 = ax2.contourf(input_surface[0,0].squeeze())
        plt.colorbar(plot2,ax=ax2)
        ax2.title.set_text('input_surface 0')

        ax3 = fig.add_subplot(223)
        plot3 = ax3.contourf(target[0,0,0].squeeze())
        ax3.title.set_text('target 0')
        plt.colorbar(plot3,ax=ax3)


        ax4 = fig.add_subplot(224)
        plot4 = ax4.contourf(target_surface[0,0].squeeze())
        ax4.title.set_text('target_surface 0')
        plt.colorbar(plot4,ax=ax4)
        plt.tight_layout()
        plt.savefig(fname='compare_{}_{}'.format(periods[0], periods[1]))
        print("image saved!")


    elapsed = time.time() - start_time
    elapsed = str(timedelta(seconds=elapsed))
    print("Elapsed [{}]".format(elapsed))
    """
