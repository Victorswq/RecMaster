from Sequential_Model.abstract_model import abstract_model
from Dataset.abstract_dataset import abstract_dataset
from Utils.utils import *
from Utils.evaluate import *
from Utils.loss_function import *


import torch
import torch.nn as nn
import torch.optim as optim


class SHAN(abstract_model):

    def __init__(self,model_name="SHAN",
                 data_name="ml-1m",
                 learning_rate=0.001,
                 embedding_size=32,
                 episodes=100,
                 seq_len=10,
                 short=1):
        super(SHAN, self).__init__(model_name=model_name,data_name=data_name)
        self.learning_rate=learning_rate
        self.episodes=episodes
        self.seq_len=seq_len
        self.short=short
        self.embedding_size=embedding_size

        # build the dataset
        self.dataset=Data_for_SHAN(data_name=data_name,seq_len=self.seq_len)
        self.user_number=self.dataset.user_number
        self.item_number=self.dataset.item_number

        # build the variables
        self.build_variables()

        # build the loss function
        self.criterion=BPRLoss()

        # init the weight of module parameter
        self.apply(self.init_weights)



    def build_variables(self):

        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.long_attention=Long_Memory_Attention(embedding_size=self.embedding_size)
        self.short_attention=Short_Memory_Attention(embedding_size=self.embedding_size)


    def forward(self,data):
        seq_item,user=data
        seq_item_embedding=self.item_matrix(seq_item)
        user_embedding=self.user_matrix(user)
        short_memory=seq_item_embedding[:,-self.short:,:]

        # get the mask
        mask=seq_item.data.eq(0)

        long_memory=self.long_attention.forward(seq_item_embedding=seq_item_embedding,user_embedding=user_embedding,mask=mask)
        memory=self.short_attention.forward(long_memory=long_memory,user_embedding=user_embedding,short_embedding=short_memory)
        # batch_size * embedding_size

        return memory


    def calculate_loss(self,data):

        seq_item,user,pos_item,neg_item=data

        memory=self.forward([seq_item,user])

        pos_item_embedding=self.item_matrix(pos_item)
        pos_scores=torch.mul(memory,pos_item_embedding).sum(dim=1)

        neg_item_embedding=self.item_matrix(neg_item)
        neg_scores=torch.mul(memory,neg_item_embedding).sum(dim=1)

        loss=self.criterion.forward(pos_scores=pos_scores,neg_scores=neg_scores)

        return loss


    def get_tiem_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def prediction(self,data):

        seq_item,user=data
        memory = self.forward([seq_item, user])

        prediction=torch.matmul(memory,self.get_tiem_matrix_weight_transpose())

        return prediction


class Long_Memory_Attention(nn.Module):

    def __init__(self,embedding_size=32):
        super(Long_Memory_Attention, self).__init__()
        self.embedding_size=embedding_size
        self.w1=nn.Linear(self.embedding_size,self.embedding_size)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size)
        self.relu=nn.ReLU()


    def forward(self,seq_item_embedding,user_embedding,mask=None):
        """

        :param seq_item_embedding: batch_size * seq_len * embedding_size
        :param user_embedding: batch_size * embedding_size
        :param mask: batch_size * seq_len
        :return:
        """
        seq_item_embedding_value=seq_item_embedding
        batch_size,seq_len,embedding_size=seq_item_embedding.size()

        seq_item_embedding=self.relu(self.w1(seq_item_embedding))
        user_embedding=user_embedding.unsqueeze(1).repeat(1,seq_len,1)
        output=(seq_item_embedding+user_embedding).sum(dim=2)
        # batch_size * seq_len
        if mask is None:
            output.masked_fill_(mask,-1e9)
        output=nn.Softmax(dim=-1)(output)
        output=output.unsqueeze(2).repeat(1,1,embedding_size)
        output=torch.mul(seq_item_embedding_value,output).sum(dim=1)
        # batch_size * embedding_size

        return output


class Short_Memory_Attention(nn.Module):

    def __init__(self,embedding_size=32):
        super(Short_Memory_Attention, self).__init__()
        self.embedding_size=embedding_size
        self.w1=nn.Linear(self.embedding_size,self.embedding_size)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size)
        self.relu=nn.ReLU()


    def forward(self,long_memory,user_embedding,short_embedding):
        """

        :param long_memory: batch_size * embedding_size
        :param user_embedding: batch_size * embedding_size
        :param short_embedding: batch_size * short_len * embedding_size
        :return:
        """
        long_memory=long_memory.unsqueeze(1)
        memory=torch.cat([long_memory,short_embedding],dim=1)
        memory_value=memory

        memory=self.relu(self.w1(memory))
        batch_size,seq_len,embedding_size=memory.size()

        user_embedding=user_embedding.unsqueeze(1).repeat(1,seq_len,1)

        output=(user_embedding+memory).sum(dim=2)
        # batch_size * seq_len
        output=nn.Softmax(dim=-1)(output)
        # batch_size * seq_len
        output=output.unsqueeze(2).repeat(1,1,embedding_size)
        output=torch.mul(output,memory_value).sum(dim=1)
        # batch_size * embedding_size

        return output


class Data_for_SHAN(abstract_dataset):

    def __init__(self,seq_len=10,data_name="ml-1m",min_user_number=5,min_item_number=5):
        super(Data_for_SHAN, self).__init__(data_name=data_name,sep="\t")
        self.seq_len=seq_len
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number

        # clean the dataset
        self.clean_data(min_item_number=self.min_item_number,min_user_number=self.min_user_number)


    def get_data_for_model(self):

        data_value=self.data.values
        user_item={}

        for value in data_value:
            user_id=value[0]
            item_id=value[1]
            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
            else:
                user_item[user_id]=[item_id]

        train_data, validation_data, test_data = [], [], []

        for user,item_list in user_item.items():
            length=len(item_list)
            if length<4:continue

            test=item_list[-self.seq_len-1:]
            test+=[user]
            test_data.append(test)

            valid=item_list[-self.seq_len-2:-1]
            valid+=[user]
            validation_data.append(valid)

            for i in range(1,length-2):
                train=item_list[:i+1]
                target=train[-1]
                if len(item_list)>self.seq_len+1:
                    train=train[-self.seq_len-1:]
                train+=[user]
                neg=np.random.choice(self.item_number)
                if neg==0 or neg==target:
                    neg=np.random.choice(self.item_number)
                train+=[neg]
                train_data.append(train)

        train_data=np.array(self.pad_sequence(train_data,seq_len=self.seq_len+3))
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
            train_data: (seq_item,user,pos_item,neg_item)
            """
            self.optimizer.zero_grad()
            seq_item=train_data[:,:-3]
            user=train_data[:,-2]
            pos_item=train_data[:,-3]
            neg_item=train_data[:,-1]
            seq_item,user,pos_item,neg_item=torch.LongTensor(seq_item),torch.LongTensor(user),torch.LongTensor(pos_item),torch.LongTensor(neg_item)
            loss = self.model.calculate_loss(data=[seq_item,user,pos_item,neg_item])
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
                """
                seq_item,user
                """
                validation=validation_data
                label = validation[:, -2]
                validation=torch.LongTensor(validation)
                seq_item=validation[:,:-2]
                user=validation[:,-1]
                scores=self.model.prediction([seq_item,user])
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                Recall(actual_set,scores.detach().numpy())


model=SHAN(data_name="ml-1m",embedding_size=128,learning_rate=0.005)
trainer=trainer(model)
trainer.train()