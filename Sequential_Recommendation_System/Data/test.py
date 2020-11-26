import pandas as pd


data=pd.read_csv("ml-1m.inter",sep=",")
value=data.values
print(value.shape)