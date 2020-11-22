import time
import pandas as pd
import numpy as np


data=pd.read_csv("Diginetica.csv",sep=";")
data=data.drop("user_id",1)
eventdate=data["eventdate"].values

new=np.zeros_like(eventdate)
for idx,curdate in enumerate(eventdate):
    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    new[idx]=date

data["eventdate"]=new
data.to_csv("diginetica.inter",header=["session_id","item_id","ratings","timeframe"],index=False)
