from Sequential_Model.abstract_model import abstract_model
from Dataset.abstract_dataset import abstract_dataset
from Utils.utils import *
from Utils.evaluate import *
from Utils.loss_function import *

import torch
import torch.nn as nn
import torch.optim as optim


class MF(abstract_model):

    def __init__(self,model_name="MF",
                 data_name="ml-100k",
                 learning_rate=0.001,
                 embedding_size=32,
                 neg_numbers=1):
        super(MF, self).__init__(model_name=model_name,data_name=data_name)
        self.learning_rate=learning_rate
        self.embedding_size=embedding_size
        self.neg_numbers=neg_numbers

        # build the dataset
        self.dataset=Data_for_MF(data_name=data_name,neg_numbers=neg_numbers)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        # build the variables
        self.build_variables()

        # build the loss function
        self.criterion=BPRLoss()


    def build_variables(self):

        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)


    def forward(self,data):

        item,user=data
        item_embedding=self.item_matrix(item)
        user_embedding=self.user_matrix(user)

        return torch.mul(item_embedding,user_embedding).sum(dim=1)


    def calculate_loss(self,data):

        pos_item,user,neg_item=data

        pos_scores=self.forward([pos_item,user])
        neg_scores=self.forward([neg_item,user])

        loss=self.criterion.forward(pos_scores,neg_scores)

        return loss


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def prediction(self,data):

        user=data
        user_embedding=self.user_matrix(user)
        prediction=torch.matmul(user_embedding,self.get_item_matrix_weight_transpose())

        return prediction


class Data_for_MF(abstract_dataset):

    def __init__(self,data_name="ml_100k",neg_numbers=1,min_user_number=10,min_item_number=10,ratio=[0.8,0.1,0.1],shuffle=True):
        super(Data_for_MF, self).__init__(data_name=data_name)
        self.neg_numbers=neg_numbers
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number
        self.ratio=ratio
        self.shuffle=shuffle

        # clean the dataset
        self.clean_data(min_user_number=self.min_user_number,min_item_number=self.min_item_number)


    def get_data_for_model(self):

        data_value=self.data.values
        data=[]

        for value in data_value:
            user_id=value[0]
            item_id=value[1]
            neg=np.random.choice(self.item_number)
            while neg==0 or neg==item_id:
                neg=np.random.choice(self.item_number)
            data.append([user_id,item_id,neg])
        data=np.array(data)

        data=self.split_by_ratio(data=data,ratio=self.ratio,shuffle=self.shuffle)

        return data


class trainer():

    def __init__(self,model):
        self.model=model


    def train_epoch(self,data):
        train_data_value = data
        total_loss=0
        count=1
        for train_data in get_batch(train_data_value,batch_size=self.model.batch_size):
            """
            train_data: (seq_item,user,pos_item,neg_item)
            """
            self.optimizer.zero_grad()
            train_data=torch.LongTensor(train_data)
            user=train_data[:,0]
            pos_item=train_data[:,1]
            neg_item=train_data[:,2]
            loss = self.model.calculate_loss(data=[pos_item,user,neg_item])
            if count%500==0:
                print("the %d step  the current loss is %f"%(count,loss))
            count+=1
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss


    def train(self):
        self.model.logging()
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.model.learning_rate)
        train_data,validation_data,test_data=self.model.dataset.get_data_for_model()
        print(len(validation_data))
        for episode in range(self.model.episodes):
            loss=self.train_epoch(data=train_data)
            print("loss is ",loss)

            if (episode+1)%self.model.verbose==0:
                """
                seq_item,user
                """
                validation=validation_data
                label = validation[:, -1]
                validation=torch.LongTensor(validation)
                user=validation[:,0]
                scores=self.model.prediction(user)
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])