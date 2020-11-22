import pandas as pd


data=pd.read_csv("ratings.dat",sep="::",engine="python")
data.to_csv("ml-1m.inter",header=["session_id","item_id","ratings","timeframe"],index=False)