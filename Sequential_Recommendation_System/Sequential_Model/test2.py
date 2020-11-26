import numpy as np


a=np.array(np.random.randint(low=1,high=5,size=(3,4)))
print(a)

for x in a:
    x[0]=-111
print(a)