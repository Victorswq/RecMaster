import pandas as pd
import numpy as np
import os


class abstract_dataset():

    def __init__(self,
                 data_name="ml-100k",
                 sep=",",
                 time_index_id=4,
                 user_index_id=1,
                 item_index_id=2,):
        self.data_name=data_name
        if data_name=="ml-100k":
            sep="\t"
        self.sep=sep
        self.time_index_id=time_index_id-1
        self.user_index_id=user_index_id-1
        self.item_index_id=item_index_id-1
        self.data_details={}

        self.read_data_with_data_name()


    def get_data_for_model(self):

        raise NotImplementedError


    def read_data_with_data_name(self):

        last_last_road = os.path.abspath('..')
        self.read_data_name = last_last_road+"\Data"+'\\'+self.data_name + ".inter"
        self.data=pd.read_csv(self.read_data_name,sep=self.sep)
        self.column_value=self.data.columns.values
        self.sort(by=[self.column_value[self.item_index_id],self.column_value[self.user_index_id]])


    def sort(self,by,ascending=True):

        self.data.sort_values(by=by,ascending=ascending,inplace=True,ignore_index=True)


    def pad_sequence(self,data,seq_len):

        new_data=[]
        for value in data:
            length=len(value)
            if length>seq_len:
                value=value[-seq_len:]
            else:
                while length<seq_len:
                    length+=1
                    value=[0]+value
            new_data.append(value)

        return new_data


    def clean_data(self,min_user_number=None,min_item_number=None,another_index=None,user_item_index_id=True):

        if min_user_number is not None and min_item_number is not None:
            for i in range(10):
                self.clean_min_user_inter_number(min_user_number=min_user_number)
                self.clean_min_item_inter_number(min_item_number=min_item_number)

        elif min_user_number is not None:
            self.clean_min_user_inter_number(min_user_number=min_user_number)

        elif min_item_number is not None:
            self.clean_min_item_inter_number(min_item_number=min_item_number)

        self.unique_data_id(another_index=another_index,user_item_index_id=user_item_index_id)


    def unique_data_id(self,another_index=None,user_item_index_id=True):

        if user_item_index_id is True:
            self.user_number=self.unique_id(index=self.user_index_id)
            self.item_number=self.unique_id(index=self.item_index_id)

        if another_index is not None:
            for index in another_index:
                self.unique_id(index=index)


    def unique_id(self,index=None):
        """

        :param index: like 0 or 1 or 2 , the index should start from 0
        :return:
        """
        if index is not None:
            column_value=self.data[self.column_value[index]].values
            remember_dict={}
            id_start=1
            new_value=np.ones_like(column_value)
            for idx,value in enumerate(column_value):
                if value in remember_dict.keys():
                    new_value[idx]=remember_dict[value]
                else:
                    remember_dict[value]=id_start
                    id_start+=1
                    new_value[idx]=remember_dict[value]
            self.data_details[self.column_value[index]]=id_start
            self.data[self.column_value[index]]=new_value

            return id_start


    def clean_min_user_inter_number(self,min_user_number=None):

        if min_user_number is not None:
            data_value=self.data.values
            user_item={}
            for idx,value in enumerate(data_value):
                user_id=value[self.user_index_id]
                if user_id in user_item.keys():
                    user_item[user_id]+=[idx]
                else:
                    user_item[user_id]=[idx]

            drop_index=[]
            for key,value in user_item.items():
                if len(value)<min_user_number:
                    drop_index+=value
            self.data=self.data.drop(drop_index,0)
            self.data=self.data.reset_index(drop=True)


    def clean_min_item_inter_number(self,min_item_number=None):

        if min_item_number is not None:
            data_value=self.data.values
            item_user={}
            for idx,value in enumerate(data_value):
                item_id=value[self.item_index_id]
                if item_id in item_user.keys():
                    item_user[item_id]+=[idx]
                else:
                    item_user[item_id]=[idx]

            drop_index=[]
            for key,value in item_user.items():
                if len(value)<min_item_number:
                    drop_index+=value

            self.data=self.data.drop(drop_index,0)
            self.data=self.data.reset_index(drop=True)

    def split_by_ratio(self, data, ratio=[0.8, 0.1, 0.1], shuffle=False):
        data_value = data
        length = len(data_value)

        if shuffle is True:
            random_index = np.arange(length)
            data_value = data_value[random_index]

        train_data_length = int(length * ratio[0])
        train_data = data_value[:train_data_length]

        valid_data_length = int(length * ratio[1])
        valid_data = data_value[train_data_length:valid_data_length + train_data_length]

        test_data = data_value[valid_data_length:]

        data = [train_data, valid_data, test_data]

        return data


    def leave_one_out(self,one=1,seq_len=10,neg_number=None):

        data_values=self.data.values
        user_item={}

        for value in data_values:
            user_id=value[self.user_index_id]
            item_id=value[self.item_index_id]
            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
            else:
                user_item[user_id]=[item_id]

        train_data,validation_data,test_data=[],[],[]
        for user_id,item_list in user_item.items():
            if len(item_list)<=one*3:
                continue

            test=item_list[-seq_len-one:]+[user_id]
            test_data.append(test)

            valid=item_list[-seq_len-2*one:-one]+[user_id]
            validation_data.append(valid)

            stop=len(item_list)-one*2
            while stop>one:
                train=item_list[:stop]
                if len(train)>seq_len+one:
                    train=train[-seq_len-one:]
                train+=[user_id]
                if neg_number is not None:
                    for i in range(neg_number):
                        neg=np.random.choice(self.item_number)
                        if neg==0 or neg==train[seq_len]:
                            neg=np.random.choice(self.item_number)
                        train+=[neg]
                train_data.append(train)
                stop-=1

        # padding the data with the same length
        if neg_number is not None:
            train_data = np.array(self.pad_sequence(train_data, seq_len=seq_len + one + 1+neg_number), dtype=np.int)
        else:
            train_data=np.array(self.pad_sequence(train_data,seq_len=seq_len+one+1),dtype=np.int)
        validation_data=np.array(self.pad_sequence(validation_data,seq_len=seq_len+one+1),dtype=np.int)
        test_data=np.array(self.pad_sequence(test_data,seq_len=seq_len+one+1),dtype=np.int)

        return [train_data,validation_data,test_data]