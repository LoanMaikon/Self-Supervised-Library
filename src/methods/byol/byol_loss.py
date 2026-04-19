import torch.nn.functional as F
import torch.nn as nn

class byol_loss(nn.Module):
    def __init__(self):
        super(byol_loss, self).__init__()

    def forward(self, online_feats_1, online_feats_2, target_feats_1, target_feats_2):
        online_feats_1 = F.normalize(online_feats_1, dim=1)
        online_feats_2 = F.normalize(online_feats_2, dim=1)
        target_feats_1 = F.normalize(target_feats_1, dim=1)
        target_feats_2 = F.normalize(target_feats_2, dim=1)

        loss1 = 2 - 2 * (online_feats_1 * target_feats_2.detach()).sum(dim=-1)
        loss2 = 2 - 2 * (online_feats_2 * target_feats_1.detach()).sum(dim=-1)

        return (loss1 + loss2).mean()
    