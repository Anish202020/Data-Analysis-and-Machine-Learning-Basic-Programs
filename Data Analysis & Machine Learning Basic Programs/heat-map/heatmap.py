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

couple_columns = df[['VRM','VYM','VBM']]
couple_columns.head()

phase_1_2 = couple_columns.groupby(['VYM','VBM'])


phase_1_2.plot('VYM', 'VBM','Time').head()

plt.figure(figsize=(9,9))
pivot_table = phase_1_2.pivot('VYM', 'VBM','Time')
plt.xlabel('helix 2 phase', size = 15)
plt.ylabel('helix1 phase', size = 15)
plt.title('Energy from Helix Phase Angles', size = 15)
sns.heatmap(pivot_table, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r');