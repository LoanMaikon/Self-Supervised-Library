import torch
import torch.nn as nn

class LARS(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.0,
        weight_decay=0.0,
        momentum=0.9,
        trust_coefficient=0.001,
        decay_bias=False,
        decay_norm=False,
        adapt_bias=False,
        adapt_norm=False,
        clip=False,
        eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient,
            decay_bias=decay_bias,
            decay_norm=decay_norm,
            adapt_bias=adapt_bias,
            adapt_norm=adapt_norm,
            clip=clip,
            eps=eps,
            is_bias=False,
            is_norm=False,
        )

        super().__init__(params, defaults)

    @staticmethod
    def _should_apply_weight_decay(group):
        is_bias = group.get("is_bias", False)
        is_norm = group.get("is_norm", False)

        if is_bias and not group["decay_bias"]:
            return False

        if is_norm and not group["decay_norm"]:
            return False

        return True

    @staticmethod
    def _should_apply_lars_adaptation(group):
        is_bias = group.get("is_bias", False)
        is_norm = group.get("is_norm", False)

        if is_bias and not group["adapt_bias"]:
            return False

        if is_norm and not group["adapt_norm"]:
            return False

        return True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            apply_weight_decay = self._should_apply_weight_decay(group)
            apply_lars_adaptation = self._should_apply_lars_adaptation(group)

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients")

                dp = p.grad

                if weight_decay != 0 and apply_weight_decay:
                    dp = dp.add(p, alpha=weight_decay)

                if apply_lars_adaptation:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)

                    one = torch.ones_like(param_norm)

                    q = torch.where(
                        param_norm > 0,
                        torch.where(
                            update_norm > 0,
                            group["trust_coefficient"]
                            * param_norm
                            / (update_norm + group["eps"]),
                            one,
                        ),
                        one,
                    )

                    if group["clip"] and lr > 0:
                        q = torch.minimum(q / lr, one)

                    dp = dp.mul(q)

                state = self.state[p]

                if "mu" not in state:
                    state["mu"] = torch.zeros_like(p)

                mu = state["mu"]
                mu.mul_(momentum).add_(dp)

                p.add_(mu, alpha=-lr)

        return loss
