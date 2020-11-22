import torch
import torch.nn as nn


class Dice(nn.Module):

    def __init__(self,embedding_size):

        super(Dice, self).__init__()

        self.sigmoid=nn.Sigmoid()
        self.alpha=torch.zeros((embedding_size,))


    def forward(self,score):

        score_p=self.sigmoid(score)

        return self.alpha*(1-score_p)*score+score_p*score