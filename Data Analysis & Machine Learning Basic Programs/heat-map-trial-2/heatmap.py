import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('heatmap.csv')
L = len(df)
print(L)

df['Time']=pd.Categorical(df['Time'],df.Time.unique())
df.head()

X = df.pivot("Time","VRM","VBM")
fig = plt.figure(figsize=(25,10))
r = sns.heatmap(X,cmap='BuPu')

r.set_title("Heat Map")
r.imshow(X)