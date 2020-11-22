import torch
import torch.nn as nn


class BPRLoss(nn.Module):

    def __init__(self,gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma=gamma


    def forward(self,pos_scores,neg_scores):
        loss=-torch.log(self.gamma+torch.sigmoid(pos_scores-neg_scores)).mean()

        return loss