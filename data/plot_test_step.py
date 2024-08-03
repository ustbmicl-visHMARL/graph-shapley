import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.family'] = 'SimSun'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False
map_size = 40

if map_size == 40:
    cut_step = 300
elif map_size == 60:
    cut_step = 1000
elif map_size == 80:
    cut_step = 2100
elif map_size == 100:
    cut_step = 4000

# test_result0 = pd.read_csv("../data/test_result/" + str(map_size) + "_Others.csv")
test_result1 = pd.read_csv("../data/test_result/" + str(map_size) + "_A2C_GCN.csv")
test_result2 = pd.read_csv("../data/test_result/" + str(map_size) + "_DQN_GCN.csv")

# test_result0['Category'] = pd.Categorical(test_result0['Category'],
#                                           ["Nearest Frontier", "Random", "EM", "Supervised+GCN"])
# test_result0 = test_result0.sort_values('Category')

test_result = pd.concat([ test_result1, test_result2], ignore_index=True)#test_result0,

indexNames = test_result[test_result['Timestep'] > cut_step].index
test_result.drop(indexNames, inplace=True)

# if map_size == 40:
#     lo1 = 0
#     lo2 = 3
#     lo3 = 0
# else:
#     lo1 = 1
#     lo2 = 2
#     lo3 = 2

flatui = ["#228B22", "#9932CC"]#"#b15928", "#ff7f00", "#33a02c", "#825f87", 

sns.set(style="darkgrid", color_codes=True, font="Times New Roman", font_scale=2.5)
sns.set_palette(sns.color_palette(flatui))
f1 = plt.figure(1, figsize=(10.4, 6))
sns.lineplot(x='Timestep', y='Information Entropy Reduction',linewidth =3.5, hue='Category', data=test_result)
plt.legend(bbox_to_anchor=(0.95, 0.95), loc='upper right', borderaxespad=0.)
f1.tight_layout(pad=0)

sns.set(style="darkgrid", color_codes=True, font="Times New Roman", font_scale=2.5)
sns.set_palette(sns.color_palette(flatui))
f2 = plt.figure(2,figsize=(10.4, 6))
if map_size > 40:
    plt.ylim((0.4, 1.9))
sns.lineplot(x='Timestep', y='Average Error of Landmarks',linewidth =3.5,  hue='Category', data=test_result)
plt.legend(bbox_to_anchor=(0.95, 0.95), loc='upper right', borderaxespad=0.)
f2.tight_layout(pad=0)

sns.set(style="ticks", color_codes=True, font="sans-serif", font_scale=1.5)
sns.set_palette(sns.color_palette(flatui))
f3 = plt.figure(3)  # [6.4, 4.8]
if map_size > 40:
    plt.ylim((-0.2, 27))
sns.lineplot(x='Timestep', y='Max localization uncertainty', hue='Category', data=test_result, legend=False)
f3.tight_layout(pad=0)
plt.show()
