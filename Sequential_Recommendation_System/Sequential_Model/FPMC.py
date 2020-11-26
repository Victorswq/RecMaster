from Sequential_Model.abstract_model import abstract_model
from Dataset.sequential_abstract_dataset import abstract_dataset
from Utils.evaluate import *
from Utils.utils import *

import torch
import torch.nn as nn
import torch.optim as optim


class FPMC(abstract_model):

    def __init__(self,model_name="FPMC",
                 data_name="m1-100k",
                 embedding_size=32,
                 learning_rate=0.001,
                 verbose=5):
        super(FPMC, self).__init__(model_name=model_name,data_name=data_name)
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate

        # build the dataset
        self.dataset=Data_for_FPMC(data_name=self.data_name)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number
        self.verbose=verbose

        # build the variables
        self.build_variables()

        # build the loss function
        self.criterion=BPRLoss()

        # init the weight
        self.apply(self.init_weights)


    def build_variables(self):

        self.item_matrix_for_user=nn.Embedding(self.item_number,self.embedding_size)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.item_matrix_for_last_item=nn.Embedding(self.item_number,self.embedding_size)
        self.last_item_matrix=nn.Embedding(self.item_number,self.embedding_size)


    def forward(self,data):

        last_item,user=data

        last_item_embedding=self.last_item_matrix(last_item)
        # batch_size * embedding_size
        user_embedding=self.user_matrix(user)
        # batch_size * embedding_size

        return last_item_embedding,user_embedding


    def calculate_loss(self,data):

        last_item,user,pos_item,neg_item=data

        last_item_embedding,user_embedding=self.forward([last_item,user])

        pos_item_embedding_for_last_item=self.item_matrix_for_last_item(pos_item)
        pos_item_embedding_for_user=self.item_matrix_for_user(pos_item)
        pos_scores=torch.add(torch.mul(last_item_embedding,pos_item_embedding_for_last_item),torch.mul(user_embedding,pos_item_embedding_for_user)).sum(dim=1)

        neg_item_embedding_for_last_item=self.item_matrix_for_last_item(neg_item)
        neg_item_embedding_for_user=self.item_matrix_for_user(neg_item)
        neg_scores=torch.add(torch.mul(last_item_embedding,neg_item_embedding_for_last_item),torch.mul(user_embedding,neg_item_embedding_for_user)).sum(dim=1)

        loss=self.criterion.forward(pos_scores,neg_scores)

        return loss


    def get_item_matrix_for_user_weight_transpose(self):

        return self.item_matrix_for_user.weight.t()


    def get_item_matrix_for_last_item_weight_transpose(self):

        return self.item_matrix_for_last_item.weight.t()


    def prediction(self,data):

        last_item,user=data
        last_item_embedding,user_embedding=self.forward([last_item,user])

        last_item_item=torch.matmul(last_item_embedding,self.get_item_matrix_for_last_item_weight_transpose())
        # batch_size * num_items

        user_item_item=torch.matmul(user_embedding,self.get_item_matrix_for_user_weight_transpose())
        # batch_size * num_user

        prediction=last_item_item+user_item_item

        return prediction


class BPRLoss(nn.Module):

    def __init__(self,gamma=1e-10):

        super(BPRLoss, self).__init__()
        self.gamma=gamma


    def forward(self,pos_scores,neg_scores):

        loss=-torch.log(self.gamma+torch.sigmoid(pos_scores-neg_scores)).mean()

        return loss


class Data_for_FPMC(abstract_dataset):

    def __init__(self,data_name="ml-100k",
                 min_user_number=5,
                 min_item_number=5,):
        super(Data_for_FPMC, self).__init__(data_name=data_name)

        self.clean_data(min_user_number=min_user_number,min_item_number=min_item_number)


    def get_data_for_model(self):

        data_values=self.data.values
        user_item={}

        for value in data_values:
            user_id=value[0]
            item_id=value[1]
            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
            else:
                user_item[user_id]=[item_id]

        train_data,validation_data,test_data=[],[],[]

        for user_id,item_list in user_item.items():
            length=len(item_list)
            if length<4:
                continue

            test=item_list[-2:]+[user_id]
            test_data.append(test)

            valid=item_list[-3:-1]+[user_id]
            validation_data.append(valid)

            for i in range(1,length-2):
                train=item_list[i-1:i+1]+[user_id]
                target=train[1]
                neg=np.random.choice(self.item_number)
                while neg==0 or neg==target:
                    neg=np.random.choice(self.item_number)
                train+=[neg]
                train_data.append(train)

        train_data=np.array(train_data)
        validation_data=np.array(validation_data)
        test_data=np.array(test_data)

        return [train_data,validation_data,test_data]


class trainer():

    def __init__(self,model):
        self.model=model


    def train_epoch(self,data):
        train_data_value = data
        total_loss=0
        count=1
        print(len(train_data_value))
        for train_data in get_batch(train_data_value,batch_size=self.model.batch_size):
            """
            train_data: (last_item,user,pos_item,neg_item)
            """
            self.optimizer.zero_grad()
            last_item=train_data[:,0]
            user=train_data[:,2]
            pos_item=train_data[:,1]
            neg_item=train_data[:,3]
            last_item,user,pos_item,neg_item=torch.LongTensor(last_item),torch.LongTensor(user),torch.LongTensor(pos_item),torch.LongTensor(neg_item)
            loss = self.model.calculate_loss(data=[last_item,user,pos_item,neg_item])
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
                last_item,user
                """
                validation=validation_data
                label = validation[:, 1]
                validation=torch.LongTensor(validation)
                last_item=validation[:,0]
                user=validation[:,2]
                scores=self.model.prediction([last_item,user])
                results=[]
                results+=HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])
                results+=MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])
                for result in results:
                    self.model.logger.info(result)


model=FPMC(data_name="ml-1m",learning_rate=0.001)
trainer=trainer(model)
trainer.train()