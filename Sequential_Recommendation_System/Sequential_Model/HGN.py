from Dataset.sequential_abstract_dataset import abstract_dataset
from Utils.evaluate import *
from Utils.loss_function import *
from Utils.utils import *
from Sequential_Model.abstract_model import abstract_model

import torch
import torch.nn as nn
import torch.optim as optim


class HGN(abstract_model):

    def __init__(self,model_name="HGN",
                 data_name="ml-1m",
                 embedding_size=32,
                 learning_rate=0.001,
                 verbose=1,
                 episodes=100,
                 batch_size=512,
                 seq_len=10):
        super(HGN, self).__init__(model_name=model_name,data_name=data_name)
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.verbose=verbose
        self.episodes=episodes
        self.batch_size=batch_size
        self.seq_len=seq_len

        # build the dataset
        self.dataset=Data_for_HGN(data_name=self.data_name,seq_len=self.seq_len)
        self.user_number=self.dataset.user_number
        self.item_number=self.dataset.item_number

        # build the variables
        self.build_variables()

        # build the loss function
        self.criterion=BPRLoss()

        # init the weight of the parameter
        self.apply(self.init_weights)


    def build_variables(self):

        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.feature_gating=Feature_Gating(embedding_size=self.embedding_size)
        self.instance_gating=Instance_Gating(embedding_size=self.embedding_size,seq_len=self.seq_len)


    def forward(self,data):

        seq_item,user=data
        seq_item_embedding=self.item_matrix(seq_item)
        user_embedding=self.user_matrix(user)

        item_item_product=seq_item_embedding.sum(dim=1)
        # batch_size * embedding_size
        feature_embedding=self.feature_gating.forward(seq_item_embedding=seq_item_embedding,user_embedding=user_embedding)
        # batch_size * seq_len * embedding_size
        instance_embedding=self.instance_gating.forward(user_item=feature_embedding,user_embedding=user_embedding)
        # batch_size * embedding_size

        embedding=item_item_product+instance_embedding+user_embedding
        # batch_size * embedding_size

        return embedding


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def calculate_loss(self,data):
        seq_item,user,pos_item,neg_item=data

        embedding=self.forward([seq_item,user])

        pos_item_embedding=self.item_matrix(pos_item)
        pos_scores=torch.mul(embedding,pos_item_embedding).sum(dim=1)

        neg_item_embedding=self.item_matrix(neg_item)
        neg_scores=torch.mul(embedding,neg_item_embedding).sum(dim=1)

        loss=self.criterion.forward(pos_scores=pos_scores,neg_scores=neg_scores)

        return loss


    def prediction(self,data):

        seq_item,user=data
        embedding=self.forward([seq_item,user])
        prediction=torch.matmul(embedding,self.get_item_matrix_weight_transpose())
        # batch_size * num_items

        return prediction


class Feature_Gating(nn.Module):

    def __init__(self,embedding_size=32):
        super(Feature_Gating, self).__init__()
        self.embedding_size=embedding_size
        self.w1=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.b=nn.Parameter(torch.zeros(self.embedding_size),requires_grad=True)
        self.sigmoid=nn.Sigmoid()


    def forward(self,seq_item_embedding,user_embedding):
        """

        :param seq_item_embedding: batch_size * seq_len * embedding_size
                == w1 == >> batch_size * seq_len * embedding_size
        :param user_embedding: batch_size * embedding_size
                == w2 == >> batch_size * embedding_size
                == unsqueeze & repeat == >> batch_size * seq_len * embedding_size
        ***
        user_item = seq_item_embedding + user_embedding
        :return:
        """
        batch_size,seq_len,embedding_size=seq_item_embedding.size()
        seq_item_embedding_value=seq_item_embedding

        seq_item_embedding=self.w1(seq_item_embedding)
        user_embedding=self.w2(user_embedding)
        user_embedding=user_embedding.unsqueeze(1).repeat(1,seq_len,1)

        user_item=self.sigmoid(seq_item_embedding+user_embedding+self.b)

        user_item=torch.mul(seq_item_embedding_value,user_item)

        return user_item


class Instance_Gating(nn.Module):

    def __init__(self,embedding_size=32,seq_len=10):
        super(Instance_Gating, self).__init__()
        self.embedding_size=embedding_size
        self.seq_len=seq_len
        self.w1=nn.Linear(self.embedding_size,1,bias=False)
        self.w2=nn.Linear(self.embedding_size,self.seq_len,bias=False)
        self.sigmoid=nn.Sigmoid()


    def forward(self,user_item,user_embedding):
        """

        :param user_item: batch_size * seq_len * embedding_size
                == w1 == >> batch_size * seq_len * 1
        :param user_embedding: batch_size * embedding_size
                == w2 == >> batch_size * seq_len
                == unsqueeze dim 2 == >> batch_size * seq_len * 1
        output = nn.sigmoid( user_item + user_item_embedding )
        output = output * user_item_value
        == mean dim 1 == >> batch_size * embedding_size
        :return:
        """
        user_embedding_value=user_item

        user_item=self.w1(user_item)
        user_embedding=self.w2(user_embedding).unsqueeze(2)

        instance_score=self.sigmoid(user_item+user_embedding).repeat(1,1,self.embedding_size)
        output=torch.mul(instance_score,user_embedding_value).sum(dim=1)
        # batch_size * embedding_size
        instance_score=instance_score.sum(dim=1)
        output=output/instance_score

        return output


class Data_for_HGN(abstract_dataset):

    def __init__(self,data_name="ml-1m",seq_len=10,min_user_number=5,min_item_number=5):
        super(Data_for_HGN, self).__init__(data_name=data_name)
        self.seq_len=seq_len
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number

        # clean the dataset
        # sep="\t"
        self.clean_data(min_user_number=self.min_user_number,min_item_number=self.min_item_number)


    def get_data_for_model(self):

        data_values=self.data.values

        user_item={}
        user_item[0]=[]
        for value in data_values:
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
                neg=np.random.choice(self.item_number)
                while neg==0 or neg==train[-2]:
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
                results=[]
                results+=HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                results+=MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                Recall(actual_set,scores.detach().numpy())
                for result in results:
                    self.model.logger.info(result)


model=HGN(data_name="ml-100k",learning_rate=0.001,seq_len=5)
trainer=trainer(model)
trainer.train()