from Sequential_Model.abstract_model import abstract_model
from Dataset.sequential_abstract_dataset import abstract_dataset
from Utils.evaluate import *
from Utils.utils import get_batch
import torch
import torch.nn as nn
import torch.optim as optim


class STAMP(abstract_model):

    def __init__(self,model_name="STAMP",
                 data_name="diginetica",
                 embedding_size=32,
                 learning_rate=0.001,
                 min_user_number=2,
                 min_item_number=5,):
        super(STAMP, self).__init__(model_name=model_name,data_name=data_name)
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number

        # build the dataset
        self.dataset=Data_for_STAMP(data_name=data_name,min_user_number=min_user_number,min_item_number=min_item_number)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        # build the loss_function
        self.criterion=nn.CrossEntropyLoss()

        # build variables
        self.build_variables()

        # init the weight
        self.apply(self.init_weights)


    def build_variables(self):

        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.attention=Attention(embedding_size=self.embedding_size)
        self.w1=nn.Linear(self.embedding_size,self.embedding_size)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size)


    def forward(self,data):

        seq_item=data
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size

        mt=seq_item_embedding[:,-1,:]
        # batch_size * embedding_size

        ms=torch.div(seq_item_embedding.sum(dim=1),seq_item_embedding.size(2))
        # batch_size * embedding_size

        # get the mask
        mask=seq_item.data.eq(0)

        ma=self.attention.forward(all_memory=seq_item_embedding,last_memory=mt,average_memory=ms,mask=mask)+ms

        hs=self.w1(ma)
        ht=self.w2(mt)
        # batch_size * embedding_size

        # prediction
        hst=torch.mul(hs,ht)

        return hst


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def calculate_loss(self,data):

        seq_item,label=data
        hst=self.forward(seq_item)
        # batch_size * embedding_size

        prediction=torch.matmul(hst,self.get_item_matrix_weight_transpose())
        # batch_size * num_items

        loss=self.criterion(prediction,label)

        return loss


    def prediction(self,data):

        seq_item=data
        hst=self.forward(seq_item)
        # batch_size * embedding_size

        prediction=torch.matmul(hst,self.get_item_matrix_weight_transpose())
        # batch_size * num_items

        return prediction


class Attention(nn.Module):

    def __init__(self,embedding_size=32):
        super(Attention, self).__init__()
        self.embedding_size=embedding_size
        self.w1=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.w3=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.b=nn.Parameter(torch.zeros(self.embedding_size),requires_grad=True)
        self.sigmoid=nn.Sigmoid()
        self.w0=nn.Linear(self.embedding_size,1,bias=False)


    def forward(self,all_memory,last_memory,average_memory,mask=None):
        """

        :param all_memory: batch_size * seq_len * embedding_size
            == w1 == >> batch_size * seq_len * embedding_size
        :param last_memory: batch_size * embedding_size
            == w2 == >> batch_size * embedding_size
            == unsqueeze & repeat == >> batch_size * seq_len * embedding_size
        :param average_memory: batch_size * embedding_size
            == w3 == >> batch_size * embedding_size
            == unsqueeze & repeat == >> batch_size * seq_len * embedding_size
        :param mask: batch_size * seq_len
        :return:
        """
        batch_size,seq_len,embedding_size=all_memory.size()
        all_memory_value=all_memory

        all_memory=self.w1(all_memory)
        last_memory=self.w2(last_memory)
        average_memory=self.w3(average_memory)

        last_memory=last_memory.unsqueeze(1).repeat(1,seq_len,1)
        average_memory=average_memory.unsqueeze(1).repeat(1,seq_len,1)
        output=self.sigmoid(all_memory+last_memory+average_memory+self.b)
        # batch_size * seq_len * embedding_size

        output=self.w0(output).squeeze(2)
        # batch_size * seq_len


        if mask is not None:
            output.masked_fill_(mask,0)

        # output=nn.Softmax(dim=1)(output)

        output=output.unsqueeze(2).repeat(1,1,embedding_size)
        # batch_size * seq_len * embedding_size

        output=torch.mul(output,all_memory_value).sum(dim=1)
        # batch_size * embedding_size

        return output


class Data_for_STAMP(abstract_dataset):

    def __init__(self,data_name="diginetica",
                 seq_len=6,
                 min_user_number=2,
                 min_item_number=5,):
        super(Data_for_STAMP, self).__init__(data_name=data_name)
        self.seq_len=seq_len
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number

        self.clean_data(min_item_number=self.min_item_number,min_user_number=self.min_user_number)


    def get_data_for_model(self):

        train_data,validation_data,test_data=self.split_by_days(valid_test_days=[7,0.1])
        train_data=self.just_leave_one_out(train_data,seq_len=self.seq_len)
        validation_data=self.just_leave_one_out(validation_data,seq_len=self.seq_len)
        test_data=self.just_leave_one_out(test_data,seq_len=self.seq_len)

        return train_data,validation_data,test_data


    def split_by_days(self,valid_test_days=[7,0.0001]):

        valid_day,test_day=valid_test_days
        data_values=self.data.values
        max_date=np.max(data_values[:,3])

        valid_date=max_date-24*3600*valid_day
        test_date=max_date-24*3600*test_day

        valid_index=0
        test_index=0

        self.sort(by=[self.column_value[3],self.column_value[0]],ascending=True)
        data_values=self.data.values
        for idx,value in enumerate(data_values):
            time_id=value[3]
            if time_id>=valid_date:
                valid_index=idx
                break

        for idx,value in enumerate(data_values):
            time_id=value[3]
            if time_id>=test_date:
                test_index=idx
                break

        train_data=data_values[:valid_index,:]
        validation_data=data_values[valid_index:,:]
        test_data=data_values[test_index:,:]

        return train_data,validation_data,test_data


    def just_leave_one_out(self,data,seq_len=7):

        user_item={}
        for value in data:
            user=value[0]
            item=value[1]
            if user in user_item.keys():
                user_item[user]+=[item]
            else:
                user_item[user]=[item]

        data=[]
        for user,item_list in user_item.items():
            length=len(item_list)
            if length<2:
                continue
            for i in range(1,length):
                seq=item_list[:i+1]
                data.append(seq)
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
                results+=MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])
                for result in results:
                    self.model.logger.info(result)


model=STAMP(learning_rate=0.005)
trainer=trainer(model)
trainer.train()