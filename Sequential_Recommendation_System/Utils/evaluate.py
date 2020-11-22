import numpy as np
from copy import deepcopy


def HitRatio(ratingss, pos_items, top_k=[10, 20]):
    print("rating_shape is ", ratingss.shape, " target shape is ", pos_items.shape)
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
        print("-------------------------------------------------------HR@%d:  %.3f" % (k, hr / length))



def MRR(ratingss,pos_items,top_k=[10,20]):
    print("rating_shape is ", ratingss.shape, " target shape is ", pos_items.shape)
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
        print("------------------------------------------------------MRR@%d:   %.3f"%(k,mrr/length))


def Recall(pos_items,ratingss,top_k=[10,20]):
    """

    :param pos_items: the dict of the user and item
    :param ratingss: batch_size * num_items
    :param top_k:
    :return:
    """

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
        print("----------------------------------------------------Recall@%d:   %.3f"%(k,sum_recall/true_users))