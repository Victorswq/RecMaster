import numpy as np
from copy import deepcopy


def HitRatio(ratingss, pos_items, top_k=[10, 20]):
    strings=[]
    string="rating_shape is ", ratingss.shape, " target shape is ", pos_items.shape
    strings.append(string)
    length = len(ratingss)
    print(length)
    for k in top_k:
        ratings = deepcopy(ratingss)
        hr = 0
        for idx in range(length):
            rating = ratings[idx]
            rank = np.argsort(rating)[-k:]
            for index in range(len(rank)):
                if rank[index] == pos_items[idx]:
                    hr += 1
                    break
        string="-------------------------------------------------------HR@%d:  %.3f" % (k, hr / length)
        strings.append(string)

    return strings



def MRR(ratingss,pos_items,top_k=[10,20]):
    strings=[]
    string="rating_shape is ", ratingss.shape, " target shape is ", pos_items.shape
    strings.append(string)
    length=len(ratingss)
    for k in top_k:
        ratings=deepcopy(ratingss)
        mrr=0
        for idx in range(length):
            rating=ratings[idx]
            rank=np.argsort(rating)[-k:]
            for index in range(len(rank)):
                if rank[index]==pos_items[idx]:
                    mrr+=1/(k-index)
                    break
        string="------------------------------------------------------MRR@%d:   %.3f"%(k,mrr/length)
        strings.append(string)

    return strings


def Recall(pos_items,ratingss,top_k=[10,20]):
    """

    :param pos_items: the dict of the user and item
    :param ratingss: batch_size * num_items
    :param top_k:
    :return:
    """
    strings=[]
    for k in top_k:
        sum_recall=0
        num_users=len(ratingss)
        true_users=0
        for i in range(num_users):
            act_set=set(pos_items[i+1])
            pred_set=set(np.argsort(ratingss[i])[-k:])
            if len(act_set)!=0:
                sum_recall+=len(act_set&pred_set)/float(len(act_set))
                true_users+=1
        string="----------------------------------------------------Recall@%d:   %.3f"%(k,sum_recall/true_users)
        strings.append(string)

    return strings