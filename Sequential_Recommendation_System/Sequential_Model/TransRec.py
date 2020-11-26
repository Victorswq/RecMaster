from Sequential_Model.abstract_model import abstract_model
from Dataset.sequential_abstract_dataset import abstract_dataset
from Utils.loss_function import BPRLoss
from Utils.utils import get_batch
from Utils.evaluate import HitRatio,MRR

import torch
import torch.nn as nn
import torch.optim as optim


class TransRec(abstract_model):

    def __init__(self,
                 model_name="TransRec",
                 data_name="ml-100k",
                 embedding_size=64,
                 learning_rate=0.001,
                 batch_size=512,
                 episodes=100,
                 verbose=1,
                 min_user_number=5,
                 min_item_number=5,
                 seq_len=1,
                 neg_number=1,
                 ):
        super(TransRec, self).__init__(model_name=model_name,
                                       data_name=data_name)
        self.learning_rate=learning_rate
        self.embedding_size=embedding_size
        self.batch_size=batch_size
        self.episodes=episodes
        self.verbose=verbose
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number
        self.seq_len=seq_len
        self.neg_number=neg_number

        # get the dataset
        self.dataset=Data_for_TransRec(data_name=self.data_name,seq_len=seq_len,neg_number=self.neg_number)

        # get user number and item number with the help of dataset
        self.user_number=self.dataset.user_number
        self.item_number=self.dataset.item_number

        # build the variables
        self.build_variables()

        # get the loss function
        self.criterion=BPRLoss()

        # init the weight
        self.apply(self.init_weights)


    def build_variables(self):

        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)


    def forward(self,data):

        last_item,user=data
        last_item_embedding=self.item_matrix(last_item)
        user_embedding=self.user_matrix(user)

        return last_item_embedding+user_embedding


    def calculate_loss(self,data):

        last_item,pos_item,user,neg_item=data
        user_embedding=self.forward([last_item,user])

        pos_item_embedding=self.item_matrix(pos_item)
        neg_item_embedding=self.item_matrix(neg_item)

        pos_scores=torch.mul(user_embedding,pos_item_embedding).sum(dim=1)
        neg_scores=torch.mul(user_embedding,neg_item_embedding).sum(dim=1)

        loss=self.criterion.forward(pos_scores=pos_scores,neg_scores=neg_scores)

        return loss


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def prediction(self,data):

        last_item,user=data
        user_embedding=self.forward([last_item,user])

        prediction=torch.matmul(user_embedding,self.get_item_matrix_weight_transpose())

        return prediction


class Data_for_TransRec(abstract_dataset):

    def __init__(self,
                 data_name="ml-100k",
                 min_user_number=5,
                 min_item_number=5,
                 one=1,
                 seq_len=10,
                 neg_number=1,):
        super(Data_for_TransRec, self).__init__(data_name=data_name)
        self.data_name=data_name
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number
        self.one=one
        self.seq_len=seq_len
        self.neg_number=neg_number

        # clean the dataset
        self.clean_data(min_user_number=self.min_user_number,
                        min_item_number=self.min_item_number)


    def get_data_for_model(self):

        data=self.leave_one_out(one=self.one,
                                seq_len=self.seq_len,
                                neg_number=self.neg_number)

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
            train_data: (last_item, pos_item, user_id, neg_item)
            """
            self.optimizer.zero_grad()
            last_item = train_data[:, 0]
            pos_item=train_data[:,1]
            user_id = train_data[:, 2]
            neg_item=train_data[:,3]
            last_item,pos_item,user_id,neg_item=torch.LongTensor(last_item),torch.LongTensor(pos_item),torch.LongTensor(user_id),torch.LongTensor(neg_item)
            loss = self.model.calculate_loss(data=[last_item,pos_item,user_id,neg_item])
            if count%50==0:
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
        for episode in range(self.model.episodes):
            loss=self.train_epoch(data=train_data)
            print("loss is ",loss)

            if (episode+1)%self.model.verbose==0:
                validation=validation_data
                label = validation[:, 1]
                validation=torch.LongTensor(validation)
                last_item = validation[:, 0]
                user_id=validation[:,2]
                scores=self.model.prediction([last_item,user_id])
                results=[]
                results+=HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                results+=MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                for result in results:
                    self.model.logger.info(result)


model=TransRec(learning_rate=0.001,data_name="ml-100k")
trainer=trainer(model=model)
trainer.train()