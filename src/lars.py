# Code mostly from https://github.com/facebookresearch/mae/blob/main/util/lars.py

import torch


class LARS(torch.optim.Optimizer):
    """
    exclude_bias_n_norm:
        If True, skips LARS scaling + weight decay
        for p.ndim <= 1. If False, bias and normalization parameters are also
        adapted by LARS and receive weight decay.

    clip:
        If False, behaves like LARC with clip=False:
            update = lr * local_lr * grad
        If True, behaves like LARC clipping:
            update = min(local_lr, lr) * grad
        That means scaling the
        gradient by min(local_lr / lr, 1).
    """
    def __init__(
        self,
        params,
        lr=0,
        weight_decay=0,
        momentum=0.9,
        trust_coefficient=0.001,
        exclude_bias_n_norm=True,
        clip=False,
        eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient,
            exclude_bias_n_norm=exclude_bias_n_norm,
            clip=clip,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            lr = g["lr"]

            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                use_lars = (p.ndim > 1) or (not g["exclude_bias_n_norm"])

                if use_lars:
                    if g["weight_decay"] != 0:
                        dp = dp.add(p, alpha=g["weight_decay"])

                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)

                    q = torch.where(
                        param_norm > 0.,
                        torch.where(
                            update_norm > 0.,
                            g["trust_coefficient"] * param_norm / (update_norm + g["eps"]),
                            one,
                        ),
                        one,
                    )

                    if g["clip"] and lr > 0:
                        q = torch.minimum(q / lr, one)

                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)

                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-lr)

        return loss
