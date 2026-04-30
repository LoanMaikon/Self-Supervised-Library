import math

# Code from https://github.com/facebookresearch/ijepa/blob/main/src/utils/schedulers.py with some modifications

class WarmupCosineSchedule(object):
    def __init__(self, optimizer, warmup_steps, start_lr, middle_lr, final_lr, T_max):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.middle_lr = middle_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps

        self._step = 0
        self.actual_value = start_lr

    def get_value(self):
        return self.actual_value

    def state_dict(self):
        return {
            "_step": self._step,
            "actual_value": self.actual_value,
        }

    def load_state_dict(self, state_dict):
        self._step = state_dict["_step"]
        self.actual_value = state_dict["actual_value"]

    def step(self):
        self._step += 1

        if self._step <= self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.middle_lr - self.start_lr)
        else:
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(
                self.final_lr,
                self.final_lr + (self.middle_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress))
            )

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

        self.actual_value = new_lr
        return new_lr

class CosineWDSchedule(object):
    def __init__(self, optimizer, start_wd, final_wd, T_max):
        self.optimizer = optimizer
        self.start_wd = start_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0
        self.actual_value = start_wd

    def get_value(self):
        return self.actual_value

    def state_dict(self):
        return {
            "_step": self._step,
            "actual_value": self.actual_value,
        }

    def load_state_dict(self, state_dict):
        self._step = state_dict["_step"]
        self.actual_value = state_dict["actual_value"]

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.start_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.start_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd

        self.actual_value = new_wd
        return new_wd

class EMACosineSchedule(object):
    def __init__(self, start_ema, final_ema, T_max):
        self.start_ema = start_ema
        self.final_ema = final_ema
        self.T_max = T_max
        self._step = 0
        self.actual_value = start_ema

    def get_value(self):
        return self.actual_value

    def state_dict(self):
        return {
            "_step": self._step,
            "actual_value": self.actual_value,
        }

    def load_state_dict(self, state_dict):
        self._step = state_dict["_step"]
        self.actual_value = state_dict["actual_value"]

    def step(self):
        self._step += 1
        progress = self._step / float(max(1, self.T_max))
        new_ema = self.final_ema + (self.start_ema - self.final_ema) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_ema >= self.start_ema:
            new_ema = min(self.final_ema, max(self.start_ema, new_ema))
        else:
            new_ema = max(self.final_ema, min(self.start_ema, new_ema))

        self.actual_value = new_ema
        return new_ema
    
class EMALinearSchedule(object):
    def __init__(self, start_ema, final_ema, T_max):
        self.start_ema = start_ema
        self.final_ema = final_ema
        self.T_max = T_max
        self._step = 0
        self.actual_value = start_ema

    def get_value(self):
        return self.actual_value

    def state_dict(self):
        return {
            "_step": self._step,
            "actual_value": self.actual_value,
        }

    def load_state_dict(self, state_dict):
        self._step = state_dict["_step"]
        self.actual_value = state_dict["actual_value"]

    def step(self):
        self._step += 1
        progress = self._step / float(max(1, self.T_max))
        new_ema = self.start_ema + progress * (self.final_ema - self.start_ema)

        if self.final_ema >= self.start_ema:
            new_ema = min(self.final_ema, max(self.start_ema, new_ema))
        else:
            new_ema = max(self.final_ema, min(self.start_ema, new_ema))

        self.actual_value = new_ema
        return new_ema

class LinearWarmupTemperatureSchedule(object):
    def __init__(
        self,
        start_temp,
        middle_temp,
        final_temp,
        warmup_steps,
        T_max,
    ):
        self.start_temp = start_temp
        self.middle_temp = middle_temp
        self.final_temp = final_temp
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self._step = 0
        self.actual_value = start_temp
    
    def get_value(self):
        return self.actual_value
    
    def state_dict(self):
        return {
            "_step": self._step,
            "actual_value": self.actual_value,
        }
    
    def load_state_dict(self, state_dict):
        self._step = state_dict["_step"]
        self.actual_value = state_dict["actual_value"]

    def step(self):
        self._step += 1

        if self._step <= self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_temp = self.start_temp + progress * (self.middle_temp - self.start_temp)
        else:
            cosine_steps = max(1, self.T_max - self.warmup_steps)
            progress = min(1.0, float(self._step - self.warmup_steps) / float(cosine_steps))
            new_temp = self.final_temp + (self.middle_temp - self.final_temp) * 0.5 * (1. + math.cos(math.pi * progress))

        self.actual_value = new_temp
        return new_temp
