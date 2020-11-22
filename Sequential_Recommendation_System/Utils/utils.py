import numpy as np


def get_batch(data,shuffle=False,batch_size=128):

    if shuffle is True:
        index=[i for i in range(len(data))]
        np.random.shuffle(index)
        data=data[index]

    for start in range(0,len(data),batch_size):
        end=min(start+batch_size,len(data))
        yield data[start:end]