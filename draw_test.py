from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 练习的数据：
data = np.arange(25).reshape(5, 5)
data = pd.DataFrame(data)

# 绘制热度图：
plot = sns.heatmap(data)

plt.show()
