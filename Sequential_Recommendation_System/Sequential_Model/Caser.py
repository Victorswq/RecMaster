from Dataset.abstract_dataset import abstract_dataset
from Sequential_Model.abstract_model import abstract_model
from Utils.utils import *
from Utils.evaluate import *
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.optim as optim


class Caser(abstract_model):

    def __init__(self,model_name="Caser",
                 data_name="ml-100k",
                 embedding_size=32,
                 learning_rate=0.001,
                 verbose=1,
                 L=4,
                 seq_len=10,
                 times=2):
        super(Caser, self).__init__(model_name=model_name,data_name=data_name)
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.verbose=verbose
        self.L=L
        self.seq_len=seq_len
        self.times=times

        # build the dataset
        self.dataset=Data_for_Caser(data_name=self.data_name)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        # build the variables
        self.build_variables()

        # init the model parameter
        self.apply(self.init_weights)

        # build the loss function
        self.criterion=nn.CrossEntropyLoss()


    def build_variables(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.horizontal_screen=Horizontal_Screen(embedding_size=self.embedding_size,L=self.L,times=self.times)
        self.vertical_screen=Vertical_Screen(embedding_size=self.embedding_size,seq_len=self.seq_len)
        self.w1=nn.Linear(self.embedding_size+self.times*self.L,self.embedding_size)
        self.w2=nn.Linear(2*self.embedding_size,self.embedding_size)


    def forward(self,data):
        """

        :param data: data including seq_item and user
                seq_item: batch_size * seq_len
                user: batch_size
        :return:
        """
        seq_item,user=data
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        seq_item_embedding=seq_item_embedding.unsqueeze(1)
        # batch_size * 1 * seq_len * embedding_size

        horizontal_embedding=self.horizontal_screen.forward(seq_item_embedding)
        # batch_size * times_mul_L
        vertical_embedding=self.vertical_screen.forward(seq_item_embedding)
        # batch_size * embedding_size

        horizontal_vertical_embedding=torch.cat([horizontal_embedding,vertical_embedding],dim=1)
        # batch_size * times_mul_L_plus_embedding_size

        horizontal_vertical_embedding=self.w1(horizontal_vertical_embedding)
        # batch_size * embedding_size

        user_embedding=self.user_matrix(user)
        # batch_size * embedding_size

        item_user_embedding=torch.cat([horizontal_vertical_embedding,user_embedding],dim=1)
        # batch_size * embedding_size_mul_2
        item_user_embedding=self.w2(item_user_embedding)
        # batch_size * embedding_size

        return item_user_embedding


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def calculate_loss(self,data):

        seq_item,target,user=data
        item_user_embedding=self.forward([seq_item,user])
        # batch_size * embedding_size
        prediction=torch.matmul(item_user_embedding,self.get_item_matrix_weight_transpose())
        # batch_size * num_items

        loss=self.criterion(prediction,target)

        return loss


    def prediction(self,data):

        seq_item,user=data
        item_user_embedding=self.forward([seq_item,user])
        # batch_size * embedding_size

        prediction=torch.matmul(item_user_embedding,self.get_item_matrix_weight_transpose())
        # batch_size * num_items

        return prediction


class Horizontal_Screen(nn.Module):

    def __init__(self,embedding_size=32,L=4,times=2):

        super(Horizontal_Screen, self).__init__()
        self.embedding_size=embedding_size
        self.relu=nn.ReLU()
        self.L=L
        self.times=times
        a=[]
        for i in range(1,1+self.L):
            for j in range(times):
                a+=[i]
        self.conv_h=nn.ModuleList([nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(i,self.embedding_size)) for i in a])


    def forward(self,seq_item_embedding):
        """

        :param seq_item_embedding: batch_size * 1 * seq_len * embedding_size
        :return:
        """
        out_hs=list()
        for conv in self.conv_h:
            conv_out=self.relu(conv(seq_item_embedding).squeeze(3))
            # batch_size * 1 * (seq_len-i+1）
            pool_out=F.max_pool1d(conv_out,conv_out.size(2)).squeeze(2)
            # batch_size * 1
            out_hs.append(pool_out)
        out_h=torch.cat(out_hs,dim=1)
        # batch_size * times_mul_L

        return out_h


class Vertical_Screen(nn.Module):

    def __init__(self,embedding_size=32,seq_len=10):
        super(Vertical_Screen, self).__init__()
        self.embedding_size=embedding_size
        self.seq_len=seq_len
        self.conv_v=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(self.seq_len,1))


    def forward(self,seq_item_embedding):
        """

        :param seq_item_embedding: batch_size * 1 * seq_len * embedding_size
                == conv2d (1,1,seq_len,1) == >> batch_size * 1 * 1 * embedding_size
        :return:
        """
        output=self.conv_v(seq_item_embedding)
        output=output.view(-1,self.embedding_size)
        # batch_size * embedding_size

        return output


class Data_for_Caser(abstract_dataset):

    def __init__(self,seq_len=10,data_name="ml-100k",min_user_number=0,min_item_number=0):
        super(Data_for_Caser, self).__init__(data_name=data_name,sep="\t")
        self.seq_len=seq_len
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number

        # clean the data
        self.clean_data(min_user_number=self.min_user_number,min_item_number=self.min_item_number)


    def get_data_for_model(self):

        data_value=self.data.values
        user_item={}
        user_item[0]=[]

        for value in data_value:
            user_id=value[0]
            item_id=value[1]
            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
            else:
                user_item[user_id]=[item_id]

        train_data,validation_data,test_data=[],[],[]
        for user,item_list in user_item.items():
            length=len(item_list)
            if length<4:
                continue
            test=item_list[-self.seq_len-1:]+[user]
            test_data.append(test)

            valid=item_list[-self.seq_len-2:-1]+[user]
            validation_data.append(valid)

            for i in range(1,length-2):
                train=item_list[:i+1]
                if len(train)>self.seq_len+1:
                    train=train[-self.seq_len-1:]
                train+=[user]
                train_data.append(train)

        train_data=np.array(self.pad_sequence(train_data,seq_len=self.seq_len+2))
        validation_data=np.array(self.pad_sequence(validation_data,seq_len=self.seq_len+2))
        test_data=np.array(self.pad_sequence(test_data,seq_len=self.seq_len+2))

        return [train_data,validation_data,test_data,user_item]


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
            train_data: (seq_item,target,user)
            """
            self.optimizer.zero_grad()
            seq_item=train_data[:,:-2]
            target=train_data[:,-2]
            user=train_data[:,-1]
            seq_item,target,user=torch.LongTensor(seq_item),torch.LongTensor(target),torch.LongTensor(user)
            loss = self.model.calculate_loss(data=[seq_item,target,user])
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
        train_data,validation_data,test_data,actual_set=self.model.dataset.get_data_for_model()
        print(len(validation_data))
        for episode in range(self.model.episodes):
            loss=self.train_epoch(data=train_data)
            print("loss is ",loss)

            if (episode+1)%self.model.verbose==0:
                validation=validation_data
                label = validation[:, -2]
                validation=torch.LongTensor(validation)
                seq_item=validation[:,:-2]
                user=validation[:,-1]
                scores=self.model.prediction([seq_item,user])
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])
                Recall(actual_set,scores.detach().numpy())


model=Caser(data_name="ml-100k",verbose=1,learning_rate=0.01)
trainer=trainer(model)
trainer.train()