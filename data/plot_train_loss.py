import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


matplotlib.rcParams['font.family'] = 'SimSun'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False

loss_result = pd.read_csv("../data/reward_data/data_torch_logs_DQN_GCN.csv")

sns.set(style = 'darkgrid', font="Times New Roman", font_scale=2)
x = loss_result['Step']
y = loss_result['Value']
f1 = plt.figure(1, figsize=(8.4, 6.0))
plt.plot(x, y, alpha=0.8, linewidth=3, color='lightblue')
plt.plot(x, y.rolling(window=5, center=True).mean(),
         alpha=0.6, linewidth=3, color='blue')
plt.xticks(range(0, 1050000, 100000))
plt.xlabel('Timestep')
plt.ylabel('Train/loss')
f1.tight_layout(pad=0)
# 显示图表
plt.show()