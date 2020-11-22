from Sequential_Model.abstract_model import abstract_model
from Dataset.abstract_dataset import abstract_dataset
from Utils.utils import *
from Utils.evaluate import *
from Utils.loss_function import *

import torch
import torch.nn as nn
import torch.optim as optim


class DeepFM(abstract_model):

    def __init__(self,model_name="DeepFM",
                 data_name="ml-1m",
                 embedding_size=32,
                 learning_rate=0.001,
                 feature_len=10,
                 verbose=1):
        super(DeepFM, self).__init__(data_name=data_name,model_name=model_name)
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.verbose=verbose
        self.feature_len=feature_len

        # build the dataset
        self.dataset=Data_for_DeepFM(data_name=self.data_name)
        self.feature_number=self.dataset.item_number

        # build the variables
        self.build_variables()


    def build_variables(self):

        self.feature_matrix=nn.Embedding(self.feature_number,self.embedding_size,padding_idx=0)
        self.fm=FM(feature_len=self.feature_len,embedding_size=self.embedding_size)
        self.deep=Deep(embedding_size=self.embedding_size,feature_len=self.feature_len)
        self.sigmoid=nn.Sigmoid()


    def forward(self,data):

        features_seq=data
        features_seq_embedding=self.feature_matrix(features_seq)
        # batch_size * feature_len * embedding_size
        fm=self.fm.forward(feature_seq_embedding=features_seq_embedding,features=features_seq)
        deep=self.deep.forward(feature_seq_embedding=features_seq_embedding)

        output=self.sigmoid(fm+deep)

        return output


    def calculate_loss(self,data):

        feature_seq,label=data

        prediction=self.forward(feature_seq)

        loss=torch.square(label-prediction).mean()

        return loss


    def prediction(self,data):

        feature_seq=data
        prediction=self.forward(feature_seq)

        return prediction


class FM(nn.Module):

    def __init__(self,feature_len=10,embedding_size=32):
        super(FM, self).__init__()
        self.feature_len=feature_len
        self.embedding_size=embedding_size
        self.w1=nn.Linear(self.feature_len,1)


    def forward(self,feature_seq_embedding,features):
        """

        :param feature_seq_embedding: batch_size * feature_len * embedding_size
        :param features: batch_size * feature_len
        :return:
        """
        add_square=feature_seq_embedding.sum(dim=1)
        add_square=torch.square(add_square)
        # batch_size * embedding_size

        square_add=torch.square(feature_seq_embedding)
        square_add=square_add.sum(dim=1)
        # batch_size * embedding_size

        fm_2_order=add_square-square_add
        # batch_size * embedding_size
        fm_2_order=fm_2_order.sum(dim=1)
        # batch_size * 1

        fm_1_order=self.w1(features)
        # batch_size * 1

        fm=torch.add(fm_2_order,fm_1_order)


        return fm


class Deep(nn.Module):

    def __init__(self,embedding_size=32,feature_len=10):
        super(Deep, self).__init__()
        self.embedding_size=embedding_size
        self.feature_len=feature_len
        self.w1=nn.Linear(self.embedding_size*feature_len,self.embedding_size)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size)
        self.w3=nn.Linear(self.embedding_size,1)
        self.relu=nn.ReLU()


    def forward(self,feature_seq_embedding):
        """

        :param feature_seq_embedding: batch_size * feature_len * embedding_size
        :return:
        """
        batch_size=feature_seq_embedding.size(0)
        feature_seq_embedding=feature_seq_embedding.view(batch_size,-1)
        feature_seq_embedding=self.relu(self.w1(feature_seq_embedding))
        feature_seq_embedding=self.relu(self.w2(feature_seq_embedding))
        feature_seq_embedding=self.w3(feature_seq_embedding)
        # batch_size * 1

        return feature_seq_embedding


class Data_for_DeepFM(abstract_dataset):

    def __init__(self,data_name="ml-1m"):
        super(Data_for_DeepFM, self).__init__(data_name=data_name)


    def get_data_for_model(self):

        pass