import pandas as pd


data=pd.read_csv("ml-100k.inter",sep="\t")
value=data.values

item=value[:,1]
print(len(set(item)))