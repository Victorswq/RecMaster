import time
import os
import torch
import torch.nn as nn
from Log.logger import Logger
from torch.nn.init import xavier_normal_,constant_


class abstract_model(nn.Module):

    def __init__(self,
                 model_name="TransRec",
                 data_name="ml-100k",
                 learning_rate=0.001,
                 episodes=100,
                 verbose=1,
                 embedding_size=32,
                 batch_size=512,
                 min_user_number=5,
                 min_item_number=5,
                 ):

        super(abstract_model, self).__init__()
        self.model_name=model_name
        self.data_name=data_name
        self.learning_rate=learning_rate
        self.episodes=episodes
        self.verbose=verbose
        self.embedding_size=embedding_size
        self.batch_size=batch_size
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number

        self.build_logger()


    def build_variables(self):

        raise NotImplementedError


    def forward(self,data):

        raise NotImplementedError


    def calculate_loss(self,data):

        raise NotImplementedError


    def prediction(self,data):

        raise NotImplementedError


    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module,nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data,0)


    def logging(self):
        self.logger.info("------------------"+str(self.model_name)+"--------------------")
        self.logger.info("learning_rate:"+str(self.learning_rate))
        self.logger.info("batch_size:"+str(self.batch_size))
        self.logger.info("embedding_size:"+str(self.embedding_size))
        self.logger.info("number_of_epochs:"+str(self.episodes))
        self.logger.info("verbose:"+str(self.verbose))
        self.logger.info("data_name: " + str(self.data_name))
        self.logger.info("num_user:"+str(self.user_number))
        self.logger.info("num_items:"+str(self.item_number))


    def build_logger(self):
        road = os.path.abspath('..')
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger = Logger("%s\log\%s\%s" % (road, self.model_name, str(localtime)))