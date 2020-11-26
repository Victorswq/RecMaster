from Dataset.sequential_abstract_dataset import abstract_dataset
from Sequential_Model.abstract_model import abstract_model
from Utils.utils import *
from Utils.evaluate import *
from Utils.loss_function import *

import torch
import torch.nn as nn
import torch.optim as optim


class GRU(abstract_model):

    def __init__(self, model_name="GRU",
                 data_name="ml-100k",
                 embedding_size=32,
                 hidden_size=32,
                 seq_len=5,
                 learning_rate=0.001,
                 num_blocks=1,
                 num_heads=1,
                 dropout_rate=0.2,
                 batch_size=5120):
        super(GRU, self).__init__(model_name=model_name, data_name=data_name)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.num_blocks=num_blocks
        self.num_heads=num_heads
        self.dropout_rate=dropout_rate
        self.batch_size=batch_size

        # build the  dataset
        self.dataset = Data_for_GRU(data_name=self.data_name, seq_len=self.seq_len)
        self.item_number = self.dataset.item_number
        self.user_number = self.dataset.user_number

        # build the loss function
        self.criterion = nn.CrossEntropyLoss()

        # build the variables
        self.build_variables()

        # init the parameter
        self.apply(self.init_weights)


    def build_variables(self):
        self.item_matrix = nn.Embedding(self.item_number, self.embedding_size, padding_idx=0)
        self.user_matrix = nn.Embedding(self.user_number, self.embedding_size)

        self.qkv=nn.ModuleList()
        self.ffn=nn.ModuleList()

        for i in range(self.num_blocks):
            self.qkv.append(QKV(embedding_size=self.embedding_size,n_head=self.num_heads))
            self.ffn.append(FFN(embedding_size=self.embedding_size,dropout_rate=self.dropout_rate))


    def forward(self, data):
        seq_item, user = data
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        batch_size,seq_len,embedding_size=seq_item_embedding.size()

        # get the mask
        mask=self.get_mask(seq_item)

        for qkv,ffn in zip(self.qkv,self.ffn):
            seq_item_embedding=qkv.forward(seq_item_embedding,seq_item_embedding,seq_item_embedding,mask)
            seq_item_embedding=ffn.forward(seq_item_embedding)
            # batch_size * seq_len * embedding_size

        # batch_size * embedding_size
        return seq_item_embedding[:,-1,:]


    def get_item_matrix_weight_transpose(self):
        return self.item_matrix.weight.t()

    def calculate_loss(self, data):
        seq_item, user, label = data
        last_memory = self.forward([seq_item, user])
        # batch_size * embedding_size

        logit = torch.matmul(last_memory, self.get_item_matrix_weight_transpose())
        loss = self.criterion(logit, label)

        return loss


    def get_mask(self,seq_item=None):

        if seq_item is None:
            return None

        batch_size,seq_len=seq_item.size()
        # get the size of the seq_item

        padding_mask=seq_item.data.eq(0).unsqueeze(2).repeat(1,1,seq_len)

        sequence_mask=torch.ones(size=(batch_size,seq_len,seq_len))
        # batch_size * seq_len * seq_len
        sequence_mask=torch.triu(sequence_mask,diagonal=1)

        mask=torch.gt(padding_mask+sequence_mask,0)

        return mask


    def prediction(self, data):
        seq_item, user = data
        last_memory = self.forward([seq_item, user])

        prediction = torch.matmul(last_memory, self.get_item_matrix_weight_transpose())

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


class Data_for_GRU(abstract_dataset):

    def __init__(self, data_name="ml-100k", seq_len=12, min_user_number=5, min_item_number=5):
        super(Data_for_GRU, self).__init__(data_name=data_name)
        self.seq_len = seq_len
        self.min_user_number = min_user_number
        self.min_item_number = min_item_number

        # clean the dataset
        self.clean_data(min_user_number=self.min_user_number, min_item_number=self.min_item_number)

    def get_data_for_model(self):

        # sorting the number
        self.sort(by=[self.column_value[-1], self.column_value[0]], ascending=True)

        data_value = self.data.values
        user_item = {}

        for value in data_value:
            user_id = value[0]
            item_id = value[1]

            if user_id in user_item.keys():
                user_item[user_id] += [item_id]
            else:
                user_item[user_id] = [item_id]

        train_data, validation_data, test_data = [], [], []
        for user, item_list in user_item.items():
            length = len(item_list)
            if length < 4:
                continue

            # get the test data
            test = item_list[-self.seq_len - 1:] + [user]
            test_data.append(test)

            # get the validation data
            valid = item_list[-self.seq_len - 2:-1] + [user]
            validation_data.append(valid)

            # get the train data
            for i in range(length-5, length - 2):
                train = item_list[:i + 1]
                if len(train) > self.seq_len + 1:
                    train = item_list[-self.seq_len - 1:]
                train += [user]
                train_data.append(train)

        train_data = np.array(self.pad_sequence(train_data, seq_len=self.seq_len + 2))
        validation_data = np.array(self.pad_sequence(validation_data, seq_len=self.seq_len + 2))
        test_data = np.array(self.pad_sequence(test_data, seq_len=self.seq_len + 2))

        return [train_data, validation_data, test_data]


class trainer():

    def __init__(self, model):
        self.model = model

    def train_epoch(self, data):
        train_data_value = data
        total_loss = 0
        count = 1
        print(len(train_data_value))
        for train_data in get_batch(train_data_value, batch_size=self.model.batch_size):
            """
            train_data: (seq_item,user,label)
            """
            train_data = torch.LongTensor(train_data)
            self.optimizer.zero_grad()
            seq_item = train_data[:, :-2]
            label = train_data[:, -2]
            user = train_data[:, -1]

            loss = self.model.calculate_loss(data=[seq_item, user, label])
            if count % 500 == 0:
                print("the %d step  the current loss is %f" % (count, loss))
            count += 1
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def train(self):
        self.model.logging()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model.learning_rate)
        train_data, validation_data, test_data = self.model.dataset.get_data_for_model()
        print(len(validation_data))
        for episode in range(self.model.episodes):
            loss = self.train_epoch(data=train_data)
            print("loss is ", loss)

            if (episode + 1) % self.model.verbose == 0:
                validation = validation_data
                label = validation[:, -2]
                validation = torch.LongTensor(validation)
                seq_item = validation[:, :-2]
                user = validation[:, -1]
                results=[]
                scores = self.model.prediction([seq_item, user])
                results+=HitRatio(ratingss=scores.detach().numpy(), pos_items=label, top_k=[20])
                results+=MRR(ratingss=scores.detach().numpy(), pos_items=label, top_k=[20])
                for result in results:
                    self.model.logger.info(result)


model = GRU(seq_len=50, learning_rate=0.01,num_blocks=1)
# asserting embedding_size == hidden_size
trainer = trainer(model)
trainer.train()