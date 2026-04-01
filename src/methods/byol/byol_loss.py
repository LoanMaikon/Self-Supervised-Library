import torch.nn.functional as F
import torch.nn as nn

class byol_loss(nn.Module):
    def __init__(self):
        super(byol_loss, self).__init__()

    def forward(self, online_feats, target_feats):
        online_feats = F.normalize(online_feats, dim=1)
        target_feats = F.normalize(target_feats, dim=1)

        online_views = online_feats.chunk(2, dim=0)
        target_views = target_feats.chunk(2, dim=0)

        loss1 = 2 - 2 * (online_views[0] * target_views[1].detach()).sum(dim=-1)
        loss2 = 2 - 2 * (online_views[1] * target_views[0].detach()).sum(dim=-1)

        return (loss1 + loss2).mean()
    