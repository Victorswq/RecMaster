from Sequential_Model.abstract_model import abstract_model
from Dataset.abstract_dataset import abstract_dataset
from Utils.evaluate import *
from Utils.utils import *

import torch
import torch.nn as nn
import torch.optim as optim


class SASRec(abstract_model):

    def __init__(self,model_name="SASRec",
                 data_name="ml-1m",
                 embedding_size=32,
                 learning_rate=0.001,
                 num_blocks=1,
                 n_head=1,
                 dropout_rate=0.2,
                 seq_len=50,
                 min_item_number=20,
                 min_user_number=20,
                 batch_size=128):
        super(SASRec, self).__init__(model_name=model_name,data_name=data_name)
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.num_blocks=num_blocks
        self.n_head=n_head
        self.dropout_rate=dropout_rate
        self.seq_len=seq_len
        self.min_item_number=min_item_number
        self.min_user_number=min_user_number
        self.batch_size=batch_size

        # build the dataset
        self.dataset=Data_for_SASRec(data_name=data_name,seq_len=self.seq_len,min_item_number=self.min_item_number,min_user_inter=self.min_user_number)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        # build the variable
        self.build_variables()

        # build the loss function
        self.criterion=nn.BCEWithLogitsLoss()

        # init the module parameter
        self.apply(self.init_weights)


    def build_variables(self):

        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.qkv=nn.ModuleList()
        self.ffn=nn.ModuleList()

        for i in range(self.num_blocks):
            self.qkv.append(QKV(embedding_size=self.embedding_size,n_head=self.n_head))
            self.ffn.append(FFN(embedding_size=self.embedding_size,dropout_rate=self.dropout_rate))


    def forward(self,data):

        seq_item=data
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size

        batch_size,seq_len,embedding_size=seq_item_embedding.size()

        # get the mask
        padding_mask=seq_item.data.eq(0)
        padding_mask=padding_mask.unsqueeze(2).repeat(1,1,seq_len)
        sequence_mask=torch.ones(size=(batch_size,seq_len,seq_len))
        sequence_mask=sequence_mask.triu(diagonal=1)
        mask=torch.gt((padding_mask+sequence_mask),0)

        for qkv,ffn in zip(self.qkv,self.ffn):
            seq_item_embedding=qkv.forward(seq_item_embedding,seq_item_embedding,seq_item_embedding,mask)
            seq_item_embedding=ffn.forward(seq_item_embedding)
            # batch_size * seq_len * embedding_size

        return seq_item_embedding


    def calculate_loss(self,data):
        seq_item,pos_item,neg_item=data

        seq_item_embedding=self.forward(seq_item)

        indices=torch.where(seq_item!=0)

        pos_item_embedding=self.item_matrix(pos_item)
        pos_scores=torch.mul(seq_item_embedding,pos_item_embedding).sum(dim=2)
        # batch_size * seq_len
        pos_label=torch.ones_like(pos_scores)

        loss=self.criterion(pos_scores[indices],pos_label[indices])

        neg_item_embedding=self.item_matrix(neg_item)
        neg_scores=torch.mul(seq_item_embedding,neg_item_embedding).sum(dim=2)
        # batch_size * seq_len
        neg_label=torch.zeros_like(neg_scores)
        loss+=self.criterion(neg_scores[indices],neg_label[indices])

        return loss


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def prediction(self,data):

        seq_item=data
        seq_item_embedding=self.forward(seq_item)
        # batch_size * seq_len * embedding_size

        seq_item_embedding=seq_item_embedding[:,-1,:]

        prediction=torch.matmul(seq_item_embedding,self.get_item_matrix_weight_transpose())
        # batch_size * num_items

        return prediction


class QKV(nn.Module):

    def __init__(self,embedding_size=32,n_head=1):
        super(QKV, self).__init__()
        self.embedding_size=embedding_size
        self.n_head=n_head
        self.w1=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.w3=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.w4=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.layernorm=nn.LayerNorm(self.embedding_size)


    def forward(self,embedding_q,embedding_k,embedding_v,mask=None):
        """

        :param embedding_q: batch_size * seq_len * embedding_size
        :param embedding_k: batch_size * seq_len * embedding_size
        :param embedding_v: batch_size * seq_len * embedding_size
        :param mask: batch_size * seq_len * seq_len
        :return:
            hidden_size = embedding_size / n_head
        """
        batch_size,seq_len,embedding_size=embedding_q.size()
        embedding_v=self.layernorm(embedding_v)
        residual=embedding_v

        embedding_q=self.w1(embedding_q).unsqueeze(2).view(batch_size,seq_len,self.n_head,-1).transpose(1,2)
        # batch_size * n_head * seq_len * hidden_size
        embedding_k=self.w2(embedding_k).unsqueeze(2).view(batch_size,seq_len,self.n_head,-1).transpose(1,2)
        # batch_size * n_head * seq_len * hidden_size
        embedding_v=self.w3(embedding_v).unsqueeze(2).view(batch_size,seq_len,self.n_head,-1).transpose(1,2)
        # batch_size * n_head * seq_len * hidden_size

        q_k=torch.matmul(embedding_q,embedding_k.transpose(2,3))
        # batch_size * n_head * seq_len * seq_len

        # get the mask
        if mask is not None:
            mask=mask.unsqueeze(1).repeat(1,self.n_head,1,1)
            # batch_size * n_head * seq_len * seq_len
            q_k.masked_fill_(mask,-1e9)

        q_k=nn.Softmax(dim=-1)(q_k)

        qkv=torch.matmul(q_k,embedding_v)
        # batch_size * n_head * seq_len * embedding_size
        qkv=qkv.transpose(1,2).view(batch_size,seq_len,-1)
        # batch_size * seq_len * embedding_size
        qkv=self.w4(qkv)
        # batch_size * seq_len * embedding_size
        qkv+=residual
        qkv=self.layernorm(qkv)

        return qkv


class FFN(nn.Module):

    def __init__(self,embedding_size=32,dropout_rate=0.2):
        super(FFN, self).__init__()
        self.embedding_size=embedding_size
        self.dropout_rate=dropout_rate
        self.layer_norm=nn.LayerNorm(self.embedding_size)

        # build the module
        self.conv_1=nn.Conv1d(in_channels=self.embedding_size,out_channels=self.embedding_size,kernel_size=1)
        self.dropout_1=nn.Dropout(self.dropout_rate)
        self.conv_2=nn.Conv1d(in_channels=self.embedding_size,out_channels=self.embedding_size,kernel_size=1)
        self.dropout_2=nn.Dropout(self.dropout_rate)
        self.relu=nn.ReLU()


    def forward(self,seq_item_embedding):
        """

        :param seq_item_embedding: batch_size * seq_len * embedding_size
        :return:
        """
        residual=seq_item_embedding
        seq_item_embedding=seq_item_embedding.transpose(1,2)
        # batch_size * embedding_size * seq_len
        seq_item_embedding=self.dropout_1(self.relu(self.conv_1(seq_item_embedding)))
        # batch_size * embedding_size * seq_len
        seq_item_embedding=self.dropout_2(self.relu(self.conv_2(seq_item_embedding)))
        # batch_size * embedding_size * seq_len
        seq_item_embedding=seq_item_embedding.transpose(1,2)
        seq_item_embedding+=residual
        seq_item_embedding=self.layer_norm(seq_item_embedding)

        return seq_item_embedding


class Data_for_SASRec(abstract_dataset):

    def __init__(self,seq_len=50,data_name='ml-1m',min_item_number=5,min_user_inter=2):

        super(Data_for_SASRec, self).__init__(data_name=data_name)
        self.seq_len=seq_len
        self.min_item_number=min_item_number
        self.min_user_number=min_user_inter

        # clean the data
        self.clean_data(min_item_number=self.min_item_number,min_user_number=self.min_user_number)


    def get_data_for_model(self):

        user_item={}
        data_value=self.data.values

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
            test=item_list[-self.seq_len-1:]
            test_data.append(test)
            valid=item_list[-self.seq_len-2:-1]
            validation_data.append(valid)
            for i in range(3,30):
                train=item_list[-self.seq_len-i:-2]
                train_list=item_list[-self.seq_len-i:-i]
                neg=self.generate_negatives(train_list)
                train+=neg
                train_data.append(train)

        train_data=np.array(self.pad_sequence(train_data,seq_len=self.seq_len*2+1))
        validation_data=np.array(self.pad_sequence(validation_data,seq_len=self.seq_len+1))
        test_data=np.array(self.pad_sequence(test_data,seq_len=self.seq_len+1))

        return [train_data,validation_data,test_data,user_item]


    def generate_negatives(self,pos_list):
        neg_list=[]
        length=len(pos_list)
        while length<self.seq_len:
            neg_list+=[0]
            length+=1

        for i in pos_list:
            neg=np.random.choice(self.item_number)
            while neg==0 and neg==i:
                neg=np.random.choice(self.item_number)
            neg_list.append(neg)

        return neg_list


class trainer():

    def __init__(self,model):
        self.model=model


    def train_epoch(self,data):
        train_data_value = data
        total_loss=0
        count=1
        for train_data in get_batch(train_data_value,batch_size=self.model.batch_size):
            """
            train_data: (seq_item,pos_item,neg_item)
            """
            self.optimizer.zero_grad()
            train_data=torch.LongTensor(train_data)
            seq_itme=train_data[:,:self.model.seq_len]
            pos_item=train_data[:,1:self.model.seq_len+1]
            neg_item=train_data[:,-self.model.seq_len:]
            loss = self.model.calculate_loss(data=[seq_itme,pos_item,neg_item])
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
                label = validation[:, -1]
                validation=torch.LongTensor(validation)
                seq_item=validation[:,:-1]
                scores=self.model.prediction(seq_item)
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])
                Recall(actual_set, scores.detach().numpy())


model=SASRec(data_name="ml-100k",learning_rate=0.001,seq_len=15,batch_size=128,num_blocks=1)
trainer=trainer(model=model)
trainer.train()