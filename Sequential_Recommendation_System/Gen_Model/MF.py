import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from Utils.utils import *
from Utils.evaluate import *
from Gen_Model.abstract_model import abstract_model
from Dataset.general_abstract_dataset import abstract_dataset


class MF(abstract_model):

    def __init__(self,data_name="ml-100k",
                 model_name="MF",
                 learning_rate=0.001,
                 embedding_size=64):
        super(MF, self).__init__(data_name=data_name,model_name=model_name)
        self.learning_rate=learning_rate
        self.embedding_size=embedding_size

        # load the dataset information
        self.dataset=Data_for_MF(data_name=self.data_name)
        self.user_number=self.dataset.user_number
        self.item_number=self.dataset.item_number

        # build the variables
        self.build_variables()

        # build the loss
        self.loss=nn.CrossEntropyLoss()

        # init the weight of the parameters
        self.apply(self.init_weights)


    def build_variables(self):

        self.item_embedding=nn.Embedding(self.item_number,self.embedding_size)
        self.user_embedding=nn.Embedding(self.user_number,self.embedding_size)
        self.sigmoid=nn.Sigmoid()


    def forward(self,data):

        user,item=data
        user_embedding=self.user_embedding(user)
        item_embedding=self.item_embedding(item)

        return user_embedding,item_embedding


    def calculate_loss(self,data):

        user,item,rating=data
        user_embedding,item_embedding=self.forward([user,item])
        prediction=torch.matmul(user_embedding,self.get_item_embedding_weight_transpose())
        # batch_size
        loss=self.loss.forward(prediction,rating)

        return loss


    def get_item_embedding_weight_transpose(self):

        return self.item_embedding.weight.t()


    def prediction(self,data):

        user=data
        user_embedding=self.user_embedding(user)
        # batch_size * embedding_size
        prediction=torch.matmul(user_embedding,self.get_item_embedding_weight_transpose())
        # batch_size * n_items

        return prediction


class MSE(nn.Module):

    def __init__(self):
        super(MSE, self).__init__()


    def forward(self,prediction,label):

        loss=torch.square(prediction-label).mean()

        return loss


class Data_for_MF(abstract_dataset):

    def __init__(self,data_name="ml-100k",min_item_inter=5,min_user_inter=5,ratio=[0.8,0.1,0.1],shuffle=True,num_neg=1):
        super(Data_for_MF, self).__init__(data_name=data_name)
        self.min_item_inter=min_item_inter
        self.min_user_inter=min_user_inter
        self.ratio=ratio
        self.shuffle=shuffle
        self.num_neg=num_neg
        self.clean_data(min_user_number=self.min_user_inter,min_item_number=self.min_item_inter)


    def get_data_for_model(self):

        data_values=self.data.values
        rating=np.ones(shape=(len(data_values),))
        data_values[:,2]=rating
        train_data,validation_data,test_data=self.split_by_ratio(data=data_values,ratio=self.ratio,shuffle=self.shuffle)
        new_train_data=[]
        for value in train_data:
            new_train_data.append(value)
            item_id=value[1]
            for i in range(self.num_neg):
                neg_id=np.random.choice(self.item_number)
                if neg_id==0 or neg_id==item_id:
                    neg_id=np.random.choice(self.item_number)
                new_value=deepcopy(value)
                new_value[1]=neg_id
                new_value[2]=0
                new_train_data.append(new_value)
        train_data=np.array(new_train_data)


        return train_data,validation_data,test_data


class trainer():

    def __init__(self,model):
        self.model=model


    def train_epoch(self,data):
        train_data_value = data
        total_loss=0
        count=1
        for train_data in get_batch(train_data_value,batch_size=self.model.batch_size):
            """
            train_data: (user,item,rating)
            """
            self.optimizer.zero_grad()
            item=train_data[:,1]
            rating=train_data[:,-2]
            user=train_data[:,0]
            user,item,rating=torch.LongTensor(user),torch.LongTensor(item),torch.LongTensor(rating)
            loss = self.model.calculate_loss(data=[user,item,rating])
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
                validation=validation_data
                label = validation[:, -2]
                validation=torch.LongTensor(validation)
                user=validation[:,0]
                scores=self.model.prediction(user)
                results=[]
                results+=HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                results+=MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])
                for result in results:
                    self.model.logger.info(result)


model=MF(data_name="ml-100k",learning_rate=0.001)
trainer=trainer(model)
trainer.train()