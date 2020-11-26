from Dataset.sequential_abstract_dataset import abstract_dataset
from Sequential_Model.abstract_model import abstract_model
from Utils.utils import *
from Utils.evaluate import *
from Utils.loss_function import *

import torch
import torch.nn as nn
import torch.optim as optim


class DIN(abstract_model):

    def __init__(self,model_name="DIN",
                 data_name="Tmall",
                 num_feature=3,
                 layers_size=[200,80,1]):
        super(DIN, self).__init__(model_name=model_name,data_name=data_name)
        self.num_feature=num_feature
        self.layers_size=layers_size

        # build the dataset
        self.dataset=Data_for_DIN(data_name=self.data_name)
        self.user_number=self.dataset.user_number
        self.item_number=self.dataset.item_number
        self.shopper_number=self.dataset.shopper_number
        self.cate_number=self.dataset.cate_number

        # build the loss function
        self.criterion=nn.BCELoss()

        # build the variables
        self.build_variables()

        #

        # the weight of the model parameter
        self.apply(self.init_weights)


    def build_variables(self):

        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.shopper_matrix=nn.Embedding(self.shopper_number,self.embedding_size,padding_idx=0)
        self.cate_matrix=nn.Embedding(self.cate_number,self.embedding_size,padding_idx=0)
        self.attention=Attention(embedding_size=self.embedding_size,num_feature=self.num_feature)
        self.dnn=DNN(embedding_size=self.embedding_size,num_feature=self.num_feature,layers_size=self.layers_size)


    def forward(self,data):

        seq_item,seq_shopper,seq_cate,user=data

        # get the mask
        mask=seq_item.data.eq(0)

        seq_item_embedding=self.item_matrix(seq_item).unsqueeze(2)
        seq_shopper_embedding=self.shopper_matrix(seq_shopper).unsqueeze(2)
        seq_cate_embedding=self.cate_matrix(seq_cate).unsqueeze(2)
        seq_embedding=torch.cat([seq_item_embedding,seq_shopper_embedding,seq_cate_embedding],dim=2)
        # batch_size * seq_len * 3 * embedding_size

        batch_size,seq_len,_,embedding_size=seq_embedding.size()
        seq_embedding=seq_embedding.view(batch_size,seq_len,-1)
        # batch_size * seq_len * 3_mul_embedding_size

        target_item_embedding=seq_embedding[:,-1:,:]
        # batch_size * 1 * 3_mul_embedding_size
        item_seq_embedding=seq_embedding[:,:-1,:]
        # batch_size * (seq_len-1) 3_mul_embedding_size

        user_embedding=self.user_matrix(user)
        # batch_size * embedding_size
        output=self.attention.forward(seq_item_embedding=item_seq_embedding,target_item_embedding=target_item_embedding,mask=mask)
        # batch_size * size
        embedding=torch.cat([user_embedding,output,target_item_embedding],dim=1)
        # batch_size * size
        embedding=self.dnn.forward(embedding)

        return embedding


    def calculate_loss(self,data):

        seq_item, seq_shopper, seq_cate, user, label = data
        prediction=self.forward([seq_item,seq_shopper,seq_cate,user])
        loss=self.criterion(prediction,label)

        return loss


    def prediction(self,data):

        seq_item, seq_shopper, seq_cate, user, label = data
        prediction = self.forward([seq_item, seq_shopper, seq_cate, user])

        return prediction


class DNN(nn.Module):

    def __init__(self,embedding_size=32,num_feature=3,layers_size=[200,80]):
        super(DNN, self).__init__()
        self.embedding_size=embedding_size
        self.num_feature=num_feature
        self.layers_size=layers_size
        self.layers=nn.ModuleList()
        for layer_size in layers_size:
            layer=nn.Linear((1+2*self.embedding_size*self.num_feature),layer_size)
            self.layers.apply(layer)
        self.sigmoid=nn.Sigmoid()
        self.dice=nn.Tanh()

    def forward(self,embedding):

        for layer in self.layers:
            embedding=self.tanh(layer(embedding))
        embedding=self.sigmoid(embedding)
        embedding=embedding.squeeze(-1)

        return embedding


class Attention(nn.Module):

    def __init__(self,embedding_size=32,num_feature=3):
        super(Attention, self).__init__()
        self.embedding_size=embedding_size
        self.num_feature=num_feature
        self.size=self.embedding_size*self.num_feature
        self.w1=nn.Linear(self.size,self.size)
        self.w2=nn.Linear(self.size,self.size)
        self.sigmoid=nn.Sigmoid()


    def forward(self,seq_item_embedding,target_item_embedding,mask=None):
        """

        :param seq_item_embedding: batch_size * seq_len * num_feature_mul_embedding_size
        :param target_item_embedding: batch_size * 1 * num_feature_mul_embedding_size
        :param mask: batch_size * seq_len
        :return:
        """
        # process the mask
        mask=mask[:,:-1,:]
        batch_size,seq_len,size=seq_item_embedding.size()
        seq_item_embedding_value=seq_item_embedding

        seq_item_embedding=self.w1(seq_item_embedding)
        target_item_embedding=self.w2(target_item_embedding).repeat(1,seq_len,1)
        # batch_size * seq_len * size
        embedding=self.sigmoid(seq_item_embedding+target_item_embedding).sum(dim=-1)
        # batch_size * seq_len
        if mask is not None:
            embedding.masked_fill_(mask,-1e9)
        embedding=nn.Softmax(dim=-1)(embedding)
        # batch_size * seq_len
        embedding=embedding.unsqueeze(2).repeat(1,1,size)

        output=torch.mul(embedding,seq_item_embedding).sum(dim=1)
        # batch_size * size

        return output


class Pooling(nn.Module):

    def __init__(self,embedding_size=32,pooling_type="mean"):
        super(Pooling, self).__init__()
        self.embedding_size=embedding_size
        self.pooling_type=pooling_type


    def forward(self,seq_item_embedding,mask=None):
        """

        :param seq_item_embedding: batch_size * seq_len * num_feature * embedding_size
        :param mask: batch_size * seq_len * num_feature
        :return:
        """


    def mean_pooling(self,seq_item_embedding,mask=None):
        """

        :param seq_item_embedding: batch_size * seq_len * num_feature * embedding_size
        :param mask: batch_size * seq_len * num_feature
        :return:
        """


class Data_for_DIN(abstract_dataset):

    def __init__(self,seq_len=10,data_name="Tmall"):
        super(Data_for_DIN, self).__init__(data_name=data_name)
        self.seq_len=seq_len

        self.shopper_number=0
        self.cate_number=0


    def get_data_for_model(self):

        pass