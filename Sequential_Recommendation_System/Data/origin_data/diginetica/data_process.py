import pandas as pd

data=pd.read_csv("Diginetica.csv",sep=";")
data=data.drop(["user_id"],1)
data["eventdate"]=data["timeframe"]
data.to_csv("diginetica.inter",index=False,header=["session_id","item_id","ratings","timeframe"])
print(data)