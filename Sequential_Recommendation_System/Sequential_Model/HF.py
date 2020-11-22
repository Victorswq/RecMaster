from Dataset.abstract_dataset import abstract_dataset
from Sequential_Model.abstract_model import abstract_model
from Utils.utils import *
from Utils.evaluate import *
from Utils.loss_function import *

import torch
import torch.nn as nn
import torch.optim as optim


class HF(abstract_model):
    """
    HF: Hierarchical Features
    """
    def __init__(self,model_name="HF",
                 data_name="ml-100k",
                 learning_rate=0.001,
                 seq_len=10):
        super(HF, self).__init__(model_name=model_name,data_name=data_name)
        self.learning_rate=learning_rate
        self.seq_len=seq_len

        # build the dataset
        self.dataset=Data_for_HF(data_name=data_name,seq_len=seq_len)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number
        self.cate_number=self.dataset.cate_number
        self.brand_number=self.dataset.brand_number

        # build the variables
        self.build_variables()

        # build the loss function
        self.criterion=nn.CrossEntropyLoss()

        # init the weight of the module parameter
        self.apply(self.init_weights)


    def build_variables(self):

        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.cate_matrix=nn.Embedding(self.cate_number,self.embedding_size,padding_idx=0)
        self.brand_matrix=nn.Embedding(self.brand_number,self.embedding_size,padding_idx=0)
        self.hierarchical_cate_attention=Hierarchical_Cate_Attention(embedding_size=self.embedding_size)
        self.hierarchical_brand_attention=Hierarchical_Brand_Attention(embedding_size=self.embedding_size)
        self.item_cate=Hierarchical_Cate_Attention(embedding_size=self.embedding_size)
        self.item_brand=Hierarchical_Cate_Attention(embedding_size=self.embedding_size)


    def forward(self,data):

        seq_item,seq_brand,seq_cate,user=data
        seq_item_embedding=self.item_matrix(seq_item)
        seq_cate_embedding=self.cate_matrix(seq_cate)
        seq_brand_embedding=self.brand_matrix(seq_brand)
        # batch_size * seq_len * embedding_size
        mask=seq_item.data.eq(0)
        # batch_size * seq_len
        user_embedding=self.user_matrix(user)
        # batch_size * embedding_size
        cate_embedding=self.hierarchical_cate_attention.forward(seq_cate_embedding=seq_cate_embedding,user_embedding=user_embedding,mask=mask)
        brand_embedding=self.hierarchical_brand_attention.forward(seq_brand_embedding=seq_brand_embedding,cate_embedding=user_embedding,mask=mask)
        item_cate_embedding=self.item_cate.forward(seq_cate_embedding=seq_item_embedding,user_embedding=cate_embedding,mask=mask)
        item_brand_embedding=self.item_brand.forward(seq_cate_embedding=seq_item_embedding,user_embedding=brand_embedding,mask=mask)

        return item_brand_embedding


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def calculate_loss(self,data):

        seq_item,seq_brand,seq_cate,user,pos_item=data
        item_brand_embedding=self.forward([seq_item,seq_brand,seq_cate,user])
        prediction=torch.matmul(item_brand_embedding,self.get_item_matrix_weight_transpose())
        loss=self.criterion(prediction,pos_item)

        return loss


    def prediction(self,data):
        seq_item, seq_brand, seq_cate, user, pos_item = data
        item_brand_embedding = self.forward([seq_item, seq_brand, seq_cate, user])
        prediction = torch.matmul(item_brand_embedding, self.get_item_matrix_weight_transpose())

        return prediction


class Hierarchical_Cate_Attention(nn.Module):

    def __init__(self,embedding_size=32):
        super(Hierarchical_Cate_Attention, self).__init__()
        self.embedding_size=embedding_size
        self.w1=nn.Linear(self.embedding_size,self.embedding_size)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size)
        self.sigmoid=nn.Sigmoid()


    def forward(self,seq_cate_embedding,user_embedding,mask=None):
        """

        :param seq_cate_embedding: batch_size * seq_len * embedding_size
        :param user_embedding: batch_size * embedding_size
        :param mask: batch_size * seq_len
        :return:
        """
        batch_size,seq_len,embedding_size=seq_cate_embedding.size()
        seq_cate_embedding_value=seq_cate_embedding

        seq_cate_embedding=self.w1(seq_cate_embedding)
        user_embedding=self.w2(user_embedding).unsqueeze(1).repeat(1,seq_len,1)
        # batch_size * seq_len * embedding_size
        output=self.sigmoid(seq_cate_embedding+user_embedding).sum(dim=2)
        if mask is not None:
            output.masked_fill_(mask,-1e9)
        output=nn.Softmax(dim=-1)(output)
        output=output.unsqueeze(2).repeat(1,1,embedding_size)

        output=torch.mul(output,seq_cate_embedding_value).sum(dim=1)
        # batch_size * seq_len

        return output


class Hierarchical_Brand_Attention(nn.Module):

    def __init__(self,embedding_size=32):
        super(Hierarchical_Brand_Attention, self).__init__()
        self.embedding_size=embedding_size
        self.w1=nn.Linear(self.embedding_size,self.embedding_size)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size)
        self.sigmoid=nn.Sigmoid()


    def forward(self,cate_embedding,seq_brand_embedding,mask=None):
        """

        :param cate_embedding: batch_size * embedding_size
        :param seq_brand_embedding: batch_size * seq_len * embedding_size
        :param mask: bathc_size * seq_len
        :return:
        """
        seq_brand_embedding_value=seq_brand_embedding
        batch_size,seq_len,embedding_size=seq_brand_embedding.size()

        seq_brand_embedding=self.w1(seq_brand_embedding)
        cate_embedding=self.w2(cate_embedding).unsqueeze(1).repeat(1,seq_len,1)
        output=self.sigmoid(seq_brand_embedding+cate_embedding).sum(dim=2)
        # batch_size * seq_len
        if mask is not None:
            output.masked_fill_(mask,-1e9)
        output=nn.Softmax(dim=-1)(output)
        # batch_size * seq_len
        output=output.unsqueeze(2).repeat(1,1,embedding_size)
        # batch_size * seq_len * embedding_size
        output=torch.mul(output,seq_brand_embedding_value).sum(dim=1)
        # batch_size * embedding_size

        return output


class Data_for_HF(abstract_dataset):

    def __init__(self,data_name="ml-100k",seq_len=10):
        super(Data_for_HF, self).__init__(data_name=data_name)
        self.seq_len=seq_len

        self.brand_number=0
        self.cate_number=0