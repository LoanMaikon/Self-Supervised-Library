import math

def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))

def group_uses_weight_decay(group):
    is_bias = group.get("is_bias", False)
    is_norm = group.get("is_norm", False)

    decay_bias = group.get("decay_bias", False)
    decay_norm = group.get("decay_norm", False)

    if is_bias and not decay_bias:
        return False

    if is_norm and not decay_norm:
        return False

    return True

class WarmupCosineSchedule:
    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        middle_lr,
        final_lr,
        T_max,
        param_group_filter=None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.middle_lr = middle_lr
        self.final_lr = final_lr
        self.T_max = T_max
        self.cosine_steps = max(1, T_max - warmup_steps)
        self.param_group_filter = param_group_filter

        self._step = 0
        self.actual_value = start_lr

        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        self._apply(start_lr)

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
        self._apply(self.actual_value)

    def _compute_value(self, step):
        if step <= self.warmup_steps:
            progress = step / float(max(1, self.warmup_steps))
            return self.start_lr + progress * (self.middle_lr - self.start_lr)

        progress = (step - self.warmup_steps) / float(self.cosine_steps)
        progress = _clamp(progress)

        return (
            self.final_lr
            + (self.middle_lr - self.final_lr)
            * 0.5
            * (1.0 + math.cos(math.pi * progress))
        )

    def _apply(self, lr):
        for group in self.optimizer.param_groups:
            if self.param_group_filter is not None:
                if not self.param_group_filter(group):
                    continue

            if group.get("fix_lr", False):
                group["lr"] = group["initial_lr"]
            else:
                lr_scale = group.get("lr_scale", 1.0)
                group["lr"] = lr * lr_scale

    def step(self):
        self._step += 1

        new_lr = self._compute_value(self._step)
        self._apply(new_lr)
        self.actual_value = new_lr

        return new_lr
    
class CosineWDSchedule:
    def __init__(
        self,
        optimizer,
        start_wd,
        final_wd,
        T_max,
        param_group_filter=None,
        zero_excluded=True,
    ):
        self.optimizer = optimizer
        self.start_wd = start_wd
        self.final_wd = final_wd
        self.T_max = max(1, T_max)
        self.param_group_filter = param_group_filter
        self.zero_excluded = zero_excluded

        self._step = 0
        self.actual_value = start_wd

        self._apply(start_wd)

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
        self._apply(self.actual_value)

    def _compute_value(self, step):
        progress = _clamp(step / float(self.T_max))

        new_wd = (
            self.final_wd
            + (self.start_wd - self.final_wd)
            * 0.5
            * (1.0 + math.cos(math.pi * progress))
        )

        if self.final_wd <= self.start_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        return new_wd

    def _apply(self, wd):
        for group in self.optimizer.param_groups:
            if self.param_group_filter is not None:
                if not self.param_group_filter(group):
                    continue

            if group_uses_weight_decay(group):
                group["weight_decay"] = wd
            elif self.zero_excluded:
                group["weight_decay"] = 0.0

    def step(self):
        self._step += 1

        new_wd = self._compute_value(self._step)

        self._apply(new_wd)
        self.actual_value = new_wd

        return new_wd
    
class EMACosineSchedule:
    def __init__(self, start_ema, final_ema, T_max):
        self.start_ema = start_ema
        self.final_ema = final_ema
        self.T_max = max(1, T_max)

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

    def _compute_value(self, step):
        progress = _clamp(step / float(self.T_max))

        new_ema = (
            self.final_ema
            + (self.start_ema - self.final_ema)
            * 0.5
            * (1.0 + math.cos(math.pi * progress))
        )

        if self.final_ema >= self.start_ema:
            new_ema = min(self.final_ema, max(self.start_ema, new_ema))
        else:
            new_ema = max(self.final_ema, min(self.start_ema, new_ema))

        return new_ema

    def step(self):
        self._step += 1

        new_ema = self._compute_value(self._step)
        self.actual_value = new_ema

        return new_ema

class EMALinearSchedule:
    def __init__(self, start_ema, final_ema, T_max):
        self.start_ema = start_ema
        self.final_ema = final_ema
        self.T_max = max(1, T_max)

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

    def _compute_value(self, step):
        progress = _clamp(step / float(self.T_max))

        new_ema = self.start_ema + progress * (self.final_ema - self.start_ema)

        if self.final_ema >= self.start_ema:
            new_ema = min(self.final_ema, max(self.start_ema, new_ema))
        else:
            new_ema = max(self.final_ema, min(self.start_ema, new_ema))

        return new_ema

    def step(self):
        self._step += 1

        new_ema = self._compute_value(self._step)
        self.actual_value = new_ema

        return new_ema

class LinearWarmupTemperatureSchedule:
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
        self.cosine_steps = max(1, T_max - warmup_steps)

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

    def _compute_value(self, step):
        if step <= self.warmup_steps:
            progress = step / float(max(1, self.warmup_steps))
            return self.start_temp + progress * (
                self.middle_temp - self.start_temp
            )

        progress = (step - self.warmup_steps) / float(self.cosine_steps)
        progress = _clamp(progress)

        return (
            self.final_temp
            + (self.middle_temp - self.final_temp)
            * 0.5
            * (1.0 + math.cos(math.pi * progress))
        )

    def step(self):
        self._step += 1

        new_temp = self._compute_value(self._step)
        self.actual_value = new_temp

        return new_temp
