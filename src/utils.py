from torch.distributed.nn.functional import all_gather
from time import strftime, localtime
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch
import json
import os

def write_on_log(text, output_path):
    if not is_main_process():
        return

    time = strftime("%Y-%m-%d %H:%M:%S - ", localtime())
    mode = "w" if not os.path.exists(os.path.join(output_path, "log.txt")) else "a"
    with open(os.path.join(output_path, "log.txt"), mode) as file:
        file.write(time + text + "\n")

def write_on_csv(output_path, epoch, iteration, loss, lr, wd, ema="-"):
    if not is_main_process():
        return
    
    file_path = os.path.join(output_path, "log.csv")
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a") as csv_file:
        if not file_exists:
            csv_file.write("timestamp,epoch,iteration,loss,lr,wd,ema\n")

        csv_file.write(f"{strftime('%Y-%m-%d %H:%M:%S', localtime())},{epoch},{iteration},{loss},{lr},{wd},{ema}\n")

def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0

def plot_fig(x, x_name, y, y_name, fig_name, output_path):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(fig_name)

    os.makedirs(output_path + "figs", exist_ok=True)

    plt.savefig(os.path.join(output_path + "figs", f"{fig_name}.png"))
    plt.close()

def save_json(data, output_path, file_name):
    if is_main_process():
        with open(os.path.join(output_path, f"{file_name}.json"), "w") as f:
            json.dump(data, f, indent=4)

def concat_all_gather(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    return torch.cat(all_gather(tensor), dim=0)

def recreate_csv_log(output_folder, last_epoch):
    if not is_main_process():
        return

    csv_path = os.path.join(output_folder, "log.csv")
    with open(csv_path, "r") as f:
        lines = f.readlines()

        new_lines = lines[:1] + [line for line in lines[1:] if int(line.split(",")[1]) <= last_epoch]

        with open(csv_path, "w") as f:
            f.writelines(new_lines)

def get_last_epoch(output_folder):
    last_epoch_path = os.path.join(output_folder, "last_epoch.json")
    if not os.path.exists(last_epoch_path):
        return 0
    
    with open(last_epoch_path, "r") as f:
        last_epoch_data = json.load(f)
    
    return last_epoch_data.get("last_epoch", 0)

def step_schedulers_to_epoch(epoch, steps_per_epoch, lr_scheduler, wd_scheduler, ema_scheduler=None):
    if epoch == 0:
        return

    total_steps = epoch * steps_per_epoch

    for _ in range(total_steps):
        lr_scheduler.step()
        wd_scheduler.step()
        if ema_scheduler is not None:
            ema_scheduler.step()

def load_last_values(output_folder, last_epoch):
    lr_values = []
    wd_values = []
    ema_values = []
    train_loss = []

    csv_path = os.path.join(output_folder, "log.csv")
    with open(csv_path, "r") as f:
        lines = f.readlines()[1:]
    
    for line in lines:
        epoch = line.split(",")[1]
        lr = line.split(",")[4]
        wd = line.split(",")[5]
        ema = line.split(",")[6]
        if int(epoch) <= last_epoch:
            lr_values.append(float(lr))
            wd_values.append(float(wd))

            if ema != "-":
                ema_values.append(float(ema))
    
    training_info_json = json.load(open(os.path.join(output_folder, "training_info.json"), "r"))
    train_loss.extend(training_info_json.get("train_loss", []))

    return lr_values, wd_values, ema_values, train_loss
