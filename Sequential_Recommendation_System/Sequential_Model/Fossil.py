from Dataset.abstract_dataset import abstract_dataset
from Sequential_Model.abstract_model import abstract_model
from Utils.utils import *
from Utils.evaluate import *
from Utils.loss_function import *

import torch
import torch.nn as nn
import torch.optim as optim


class Fossil(abstract_model):

    def __init__(self,model_name="Fossil",
                 data_name="ml-100k",
                 seq_len=10,
                 order=1):
        super(Fossil, self).__init__(model_name=model_name,data_name=data_name)
        self.seq_len=seq_len
        self.order=order

        # build the dataset
        self.dataset=Data_for_Fossil(data_name=self.data_name,seq_len=self.seq_len)
        self.user_number=self.dataset.user_number
        self.item_number=self.dataset.item_number

        # build the variables
        self.build_variables()

        # build the loss function
        self.criterion=BPRLoss()

        # init the weight of the module parameter
        self.apply(self.init_weights)



    def build_variables(self):

        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.high_mc=High_MC(order=self.order,embedding_size=self.embedding_size)
        self.similarity=Similarity(embedding_size=self.embedding_size)


    def forward(self,data):

        seq_item,data_len=data
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        high_mc_embedding=seq_item_embedding[:,-self.order:,:]
        # batch_size * order * embedding_size
        last_embedding=seq_item_embedding[:,-1,:]
        # batch_size * embedding_size

        # get_mask
        mask=seq_item.data.eq(0)
        mask_mc=mask[:,-self.order:]
        # batch_size * order
        high_mc=self.high_mc.forward(all_memory=high_mc_embedding,last_memory=last_embedding,mask=mask_mc)
        similarity=self.similarity.forward(all_memory=seq_item_embedding,data_len=data_len)

        output=high_mc+similarity

        return output


    def calculate_loss(self,data):

        seq_item,data_len,pos_item,neg_item=data
        output=self.forward([seq_item,data_len])

        pos_item_embedding=self.item_matrix(pos_item)
        neg_item_embedding=self.item_matrix(neg_item)
        # batch_size * embedding_size

        pos_scores=torch.mul(output,pos_item_embedding).sum(dim=1)
        neg_scores=torch.mul(output,neg_item_embedding).sum(dim=1)
        # batch_size

        loss=self.criterion.forward(pos_scores=pos_scores,neg_scores=neg_scores)

        return loss


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def prediction(self,data):

        seq_item,data_len=data
        output=self.forward([seq_item,data_len])
        prediction=torch.matmul(output,self.get_item_matrix_weight_transpose())

        return prediction


class High_MC(nn.Module):

    def __init__(self,order=1,embedding_size=32):
        super(High_MC, self).__init__()
        self.order=order
        self.embedding_size=embedding_size
        self.w1=nn.Linear(self.embedding_size,self.embedding_size)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size)
        self.tanh=nn.Tanh()
        self.w3=nn.Linear(self.embedding_size,1)


    def forward(self,all_memory,last_memory,mask=None):
        """

        :param all_memory: batch_size * order * embedding_size
        :param last_memory: batch_size * embedding_size
        :param mask: batch_size * order
        :return:
        """
        batch_size,order,embedding_size=all_memory.size()
        all_memory_value=all_memory

        all_memory=self.w1(all_memory)
        last_memory=self.w2(last_memory).unsqueeze(1).repeat(1,order,1)

        output=self.tanh(all_memory+last_memory)
        output=self.w3(output).squeeze(2)
        # batch_size * order
        if mask is not None:
            output.masked_fill_(mask,-1e9)

        output=output.unsqueeze(2)
        output=torch.mul(output,all_memory_value).sum(dim=1)
        # batch_size * embedding_size

        return output


class Similarity(nn.Module):

    def __init__(self,embedding_size=32):
        super(Similarity, self).__init__()
        self.embedding_size=embedding_size


    def forward(self,all_memory,data_len):
        """

        :param all_memory: batch_size * seq_len * embedding_size
        :param data_len: batch_size
        :return:
        """

        all_memory=all_memory.sum(dim=1)
        data_len=data_len.unsqueeze(1)
        all_memory=torch.div(all_memory,data_len)
        # batch_size * embedding_size

        return all_memory


class Data_for_Fossil(abstract_dataset):

    def __init__(self,data_name="ml-100k",seq_len=10,min_user_number=5,min_item_number=5):
        super(Data_for_Fossil, self).__init__(data_name=data_name,sep="\t")
        self.seq_len=seq_len

        self.min_user_number=min_user_number
        self.min_item_number=min_item_number

        # clean the dataset
        self.clean_data(min_user_number=self.min_user_number,min_item_number=self.min_item_number)


    def get_data_for_model(self):

        data_values=self.data.values
        user_item={}

        self.sort(by=[self.column_value[-1],self.column_value[0]],ascending=True)
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

            test=item_list[-self.seq_len-1:]
            changdu=len(test)-1
            test+=[changdu]
            test_data.append(test)

            valid=item_list[-self.seq_len-2:-1]
            changdu=len(valid)-1
            valid+=[changdu]
            validation_data.append(valid)

            for i in range(1,length-2):
                train=item_list[:i+1]
                if len(train)>self.seq_len+1:
                    train=train[-self.seq_len-1:]
                changdu=len(train)-1
                train+=[changdu]
                neg=np.random.choice(self.item_number)
                if neg==train[-2] or neg==0:
                    neg=np.random.choice(self.item_number)
                train+=[neg]
                train_data.append(train)

        train_data=np.array(self.pad_sequence(train_data,seq_len=self.seq_len+3))
        validataion_data=np.array(self.pad_sequence(validation_data,seq_len=self.seq_len+2))
        test_data=np.array(self.pad_sequence(test_data,seq_len=self.seq_len+2))

        return [train_data,validataion_data,test_data]


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
            train_data=torch.LongTensor(train_data)
            self.optimizer.zero_grad()
            seq_item=train_data[:,:-3]
            pos_item=train_data[:,-3]
            data_len=train_data[:,-2]
            neg_item=train_data[:,-1]

            loss = self.model.calculate_loss(data=[seq_item,data_len,pos_item,neg_item])
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
                data_len=validation[:,-1]
                scores=self.model.prediction([seq_item,data_len])
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20])


model=Fossil(order=2,seq_len=10)
trainer=trainer(model)
trainer.train()