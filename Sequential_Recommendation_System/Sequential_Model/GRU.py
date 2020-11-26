from Dataset.sequential_abstract_dataset import abstract_dataset
from Sequential_Model.abstract_model import abstract_model
from Utils.utils import *
from Utils.evaluate import *
from Utils.loss_function import *

import torch
import torch.nn as nn
import torch.optim as optim


class GRU(abstract_model):

    def __init__(self,model_name="GRU",
                 data_name="ml-100k",
                 embedding_size=32,
                 hidden_size=32,
                 seq_len=5,
                 learning_rate=0.001):
        super(GRU, self).__init__(model_name=model_name,data_name=data_name)
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.seq_len=seq_len
        self.learning_rate=learning_rate

        # build the  dataset
        self.dataset=Data_for_GRU(data_name=self.data_name,seq_len=self.seq_len)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        # build the loss function
        self.criterion=nn.CrossEntropyLoss()

        # build the variables
        self.build_variables()

        # init the parameter
        self.apply(self.init_weights)


    def build_variables(self):

        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.gru=nn.GRU(input_size=self.embedding_size,hidden_size=self.hidden_size,batch_first=True)


    def forward(self,data):

        seq_item,user=data
        item_embedding=self.item_matrix(seq_item[:,-1])
        user_embedding=self.user_matrix(user)
        seq_item_embedding=self.item_matrix(seq_item)
        seq_item_embedding=torch.cat([user_embedding.unsqueeze(1),seq_item_embedding],dim=1)
        # batch_size * seq_len_plus_1 * embedding_size

        all_memory,last_memory=self.gru(seq_item_embedding)
        last_memory=last_memory.squeeze(0)
        # batch_size * hidden_size
        # last_memory+=user_embedding

        x=seq_item_embedding.sum(dim=1)

        return x


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def calculate_loss(self,data):

        seq_item,user,label=data
        last_memory=self.forward([seq_item,user])
        # batch_size * embedding_size

        logit=torch.matmul(last_memory,self.get_item_matrix_weight_transpose())
        loss=self.criterion(logit,label)

        return loss


    def prediction(self,data):

        seq_item,user=data
        last_memory=self.forward([seq_item,user])

        prediction=torch.matmul(last_memory,self.get_item_matrix_weight_transpose())

        return prediction


class QKV(nn.Module):

    def __init__(self,num_heads=1,embedding_size=32):
        super(QKV, self).__init__()
        self.num_heads=num_heads
        self.embedding_size=embedding_size
        try:
            assert self.embedding_size%self.num_heads==0
        except ValueError:
            raise ValueError("make sure that hidden_size%num_heads==0")
        self.hidden_size=int(self.embedding_size/self.num_heads)

        self.w1=nn.Linear(self.embedding_size,self.embedding_size)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size)
        self.w3=nn.Linear(self.embedding_size,self.embedding_size)
        self.w4=nn.Linear(self.embedding_size,self.embedding_size)
        self.layer_norm=nn.LayerNorm(self.embedding_size)

    def forward(self,q,k,v,mask=None):
        """

        :param q: batch_size * seq_len * embedding_size
        :param k: batch_size * seq_len * embedding_size
        :param v: batch_size * seq_len * embedding_size
        :param mask: batch_size * seq_len * seq_len
        :return:
        """
        v=self.layer_norm(v)
        batch_size,seq_len,embedding_size=q.size()
        residual=v

        q=self.w1(q).unsqueeze(2).view(batch_size,seq_len,self.num_heads,-1).transpose(1,2)
        k=self.w2(k).unsqueeze(2).view(batch_size,seq_len,self.num_heads,-1).transpose(1,2)
        v=self.w3(v).unsqueeze(2).view(batch_size,seq_len,self.num_heads,-1).transpose(1,2)
        # batch_size * num_heads * seq_len * hidden_size
        qk=torch.matmul(q,k.transpose(2,3))
        # batch_size * num_heads * seq_len * seq_len

        # get the mask
        # batch_size * seq_len * seq_len ===>>> batch_size * num_heads * seq_len * seq_len
        if mask is not None:
            mask=mask.unsqueeze(1).repeat(1,self.num_heads,1,1)
            qk.masked_fill_(mask,-1e9)
        mask=nn.Softmax(dim=-1)(qk)
        output=torch.matmul(mask,v)
        # batch_size * num_heads * seq_len * hidden_size
        output=output.transpose(1,2).view(batch_size,seq_len,-1)
        output=self.w4(output)
        output+=residual
        output=self.layer_norm(output)

        return output


class FFN(nn.Module):

    def __init__(self,embedding_size=32,dropout_rate=0.2):
        super(FFN, self).__init__()
        self.embedding_size=embedding_size
        self.dropout_rate=dropout_rate
        self.conv1=nn.Conv1d(in_channels=self.embedding_size,out_channels=self.embedding_size,kernel_size=1)
        self.dropout1=nn.Dropout(self.dropout_rate)
        self.conv2=nn.Conv1d(in_channels=self.embedding_size,out_channels=self.embedding_size,kernel_size=1)
        self.dropout2=nn.Dropout(self.dropout_rate)
        self.layer_norm=nn.LayerNorm(self.embedding_size)
        self.relu=nn.ReLU()


    def forward(self,embedding_after_qkv):
        """

        :param embedding_after_qkv: batch_size * seq_len * embedding_size
        :return:
        """
        embedding_after_qkv=self.layer_norm(embedding_after_qkv)
        residual=embedding_after_qkv
        embedding_after_qkv=embedding_after_qkv.transpose(1,2)
        # batch_size * embedding_size * seq_len
        embedding_after_qkv=self.relu(self.dropout1(self.conv1(embedding_after_qkv)))
        embedding_after_qkv=self.dropout2(self.conv2(embedding_after_qkv))

        embedding_after_qkv=self.layer_norm(embedding_after_qkv+residual)

        return embedding_after_qkv


class Data_for_GRU(abstract_dataset):

    def __init__(self,data_name="ml-100k",seq_len=12,min_user_number=5,min_item_number=5):
        super(Data_for_GRU, self).__init__(data_name=data_name)
        self.seq_len=seq_len
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number

        # clean the dataset
        self.clean_data(min_user_number=self.min_user_number,min_item_number=self.min_item_number)


    def get_data_for_model(self):

        # sorting the number
        self.sort(by=[self.column_value[-1], self.column_value[0]], ascending=True)

        data_value=self.data.values
        user_item={}

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

            # get the test data
            test=item_list[-self.seq_len-1:]+[user]
            test_data.append(test)

            # get the validation data
            valid=item_list[-self.seq_len-2:-1]+[user]
            validation_data.append(valid)

            # get the train data
            for i in range(1,length-2):
                train=item_list[:i+1]
                if len(train)>self.seq_len+1:
                    train=item_list[-self.seq_len-1:]
                train+=[user]
                train_data.append(train)

        train_data=np.array(self.pad_sequence(train_data,seq_len=self.seq_len+2))
        validation_data=np.array(self.pad_sequence(validation_data,seq_len=self.seq_len+2))
        test_data=np.array(self.pad_sequence(test_data,seq_len=self.seq_len+2))

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
            train_data: (seq_item,user,label)
            """
            train_data=torch.LongTensor(train_data)
            self.optimizer.zero_grad()
            seq_item=train_data[:,:-2]
            label=train_data[:,-2]
            user=train_data[:,-1]

            loss = self.model.calculate_loss(data=[seq_item,user,label])
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
                seq_item=validation[:,:-2]
                user=validation[:,-1]
                scores=self.model.prediction([seq_item,user])
                results=[]
                results+=HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])
                results+=MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])
                for result in results:
                    self.model.logger.info(result)


model=GRU(seq_len=2,learning_rate=0.001,data_name="ml-1m")
# asserting embedding_size == hidden_size
trainer=trainer(model)
trainer.train()