import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Sequential_Model.abstract_model import abstract_model
from Dataset.sequential_abstract_dataset import abstract_dataset
from Utils.evaluate import HitRatio,MRR
from Utils.utils import get_batch
from torch.nn.init import xavier_normal_,constant_


class NARM(abstract_model):

    def __init__(self,
                 model_name="NARM",
                 data_name="diginetica",
                 embedding_size=32,
                 hidden_size=32,
                 learning_rate=0.01,
                 ):
        super(NARM, self).__init__(model_name=model_name,data_name=data_name)
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size

        # build the dataset
        self.dataset=Data_for_NARM(data_name=self.data_name)
        self.user_number=self.dataset.user_number
        self.item_number=self.dataset.item_number
        self.learning_rate=learning_rate

        # build the variable
        self.build_variables()

        # build the loss_function
        self.criterion=nn.CrossEntropyLoss()

        # init the weight of the module
        self.apply(self.init_weights)


    def build_variables(self):

        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.gru=nn.GRU(input_size=self.embedding_size,hidden_size=self.hidden_size,batch_first=True)
        self.attention=Attention(hidden_size=self.hidden_size)
        self.w1=nn.Linear(2*self.hidden_size,self.embedding_size)


    def forward(self,data):
        """

        :param data: batch_size * seq_len
        :return:
        """
        seq_item=data
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        all_memory,last_memory=self.gru(seq_item_embedding)
        last_memory=last_memory.squeeze(0)
        """
        all_memory: batch_size * seq_len * hidden_size
        last_memory: batch_size * hidden_size
        """

        # get the mask
        mask=seq_item.data.eq(0)

        weight_memory=self.attention.forward(all_memory=all_memory,last_memory=last_memory,mask=mask)

        main_and_sequence_memory=torch.cat([weight_memory,last_memory],dim=1)
        # batch_size * 2_mul_hidden_size

        main_and_sequence_memory=self.w1(main_and_sequence_memory)
        # batch_size * embedding_size

        return main_and_sequence_memory


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def calculate_loss(self,data):

        seq_item,target=data
        main_and_sequence_memory=self.forward(seq_item)
        # batch_size * embedding_size
        prediction=torch.matmul(main_and_sequence_memory,self.get_item_matrix_weight_transpose())
        # batch_size * num_items

        loss=self.criterion(prediction,target)

        return loss


    def prediction(self,data):

        seq_item=data
        main_and_sequence_memory=self.forward(seq_item)
        # batch_size * embedding_size
        prediction=torch.matmul(main_and_sequence_memory,self.get_item_matrix_weight_transpose())
        # batch_size * num_items

        return prediction


class Attention(nn.Module):

    def __init__(self,hidden_size=32):
        super(Attention, self).__init__()
        self.hidden_size=hidden_size
        self.w1=nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.w2=nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.sigmoid=nn.Sigmoid()
        self.w3=nn.Linear(self.hidden_size,1,bias=False)


    def forward(self,all_memory,last_memory,mask=None):
        """

        :param all_memory: batch_size * seq_len * hidden_size
            == w1 == >> batch_size * seq_len * hidden_size
        :param last_memory: batch_size * hidden_size
            == w2 == >> batch_size * hidden_size
            == unsqueeze & repeat ==>> batch_size * seq_len * hidden_size
        :param mask: batch_size * seq_len
        :return:
        """
        batch_size,seq_len,hidden_size=all_memory.size()
        all_memory_value=all_memory

        all_memory=self.w1(all_memory)
        last_memory=self.w2(last_memory).unsqueeze(1).repeat(1,seq_len,1)

        output=self.sigmoid(all_memory+last_memory)
        output=self.w3(output).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            output.masked_fill_(mask,0)

        # output=nn.Softmax(dim=1)(output)
        # # batch_size * seq_len

        output=output.unsqueeze(2).repeat(1,1,hidden_size)
        # batch_size * seq_len * hidden_size

        output=torch.mul(output,all_memory_value).sum(dim=1)
        # batch_size * hidden_size

        return output


class Data_for_NARM(abstract_dataset):

    def __init__(self,data_name="diginetica",seq_len=7):
        super(Data_for_NARM, self).__init__(data_name=data_name)
        self.seq_len=seq_len

        # clean the data
        self.clean_data(min_user_number=2,min_item_number=5)


    def get_data_for_model(self,valid_test_days=[7,0.001]):

        train_data,validation_data,test_data=self.split_by_time(valid_test_days=valid_test_days)
        train_data=self.just_leave_one_out(train_data,seq_len=self.seq_len)
        validation_data=self.just_leave_one_out(validation_data,seq_len=self.seq_len)
        test_data=self.just_leave_one_out(test_data,seq_len=self.seq_len)

        return [train_data,validation_data,test_data]


    def split_by_time(self,valid_test_days=[3,1]):

        valid_day,test_day=valid_test_days

        data_values=self.data.values
        max_time=np.max(data_values[:,3])

        valid_time=max_time-86400*(valid_day+test_day)
        test_time=max_time-86400*test_day
        self.sort(by=[self.column_value[3],self.column_value[0]],ascending=True)

        valid_index=0
        test_index=0

        data_values=self.data.values
        for idx,value in enumerate(data_values):
            time=value[3]
            if time>valid_time:
                valid_index=idx
                break

        for idx,value in enumerate(data_values):
            time=value[3]
            if time>test_time:
                test_index=idx
                break
        train_data=data_values[:valid_index,:]
        validation_data=data_values[valid_index:test_index,:]
        test_data=data_values[test_index:,:]
        
        return train_data,validation_data,test_data
    
    
    def just_leave_one_out(self,data,seq_len=10):
        
        user_item={}
        for value in data:
            user_id=value[0]
            item_id=value[1]
            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
            else:
                user_item[user_id]=[item_id]

        data=[]
        for user,item_list in user_item.items():
            length=len(item_list)
            for i in range(1,length):
                seq_item=item_list[:i+1]
                data.append(seq_item)

        data=self.pad_sequence(data,seq_len=seq_len+1)
        data=np.array(data)

        return data


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
            train_data: (last_item, pos_item, user_id, neg_item)
            """
            self.optimizer.zero_grad()
            seq_itme=train_data[:,:-1]
            target=train_data[:,-1]
            seq_itme,target=torch.LongTensor(seq_itme),torch.LongTensor(target)
            loss = self.model.calculate_loss(data=[seq_itme,target])
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
                LEN=len(validation_data)
                x=np.unique(np.random.choice(LEN,2000))
                validation=validation_data[x,:]
                label = validation[:, -1]
                validation=torch.LongTensor(validation)
                seq_item=validation[:,:-1]
                scores=self.model.prediction(seq_item)
                results=[]
                results+=HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])
                results+=RR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])
                for result in results:
                    self.model.logger.info(result)


model=NARM(learning_rate=0.001)
trainer=trainer(model=model)
trainer.train()