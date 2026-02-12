import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap


def heatmap(map, index):
    # sns.set()
    # map = map[0: len(xl), 0: len(yl)]
    N = map.shape[0]
    R = pd.DataFrame(map, columns=np.arange(0, N), index=np.arange(0, N))
    sns_plot = sns.heatmap(R, linewidths=0, cmap='YlOrRd', vmin=0, vmax=1, xticklabels=10, yticklabels=10)
    plt.xlabel("Exercise index", size=12)
    plt.ylabel("Exercise index", size=12)
    # sns_plot = sns.heatmap(R, linewidths=0, cmap='coolwarm', vmin=0, vmax=1)
    # sns_plot.tick_params(labelsize=12, direction='in')
    # cax = plt.gcf().axes[-1]
    # cax.tick_params(labelsize=12, direction='in', top='off', bottom='off', left='off', right='off')
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=0)
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig('heatmaps/heatmap'+str(index)+'.pdf')
    plt.show()
