import scipy.io
import numpy as np
import pandas as pd

# 加载.mat文件
mat_data = scipy.io.loadmat('GM01.mat')

# 假设你想提取键名为 'data_key' 的数据，替换为实际的键名
data = mat_data['img']  # 这是一个168x112x108的数组

map = mat_data['map']
# 合并前两维
reshaped_data1 = data.reshape(-1, data.shape[2])  # 将168x112变为一个长向量，保留108维
reshaped_data2 = map.reshape(-1, 1)  # 将data2展开为一个列向量

final_data = np.hstack((reshaped_data1, reshaped_data2))

# 保存为CSV文件
df = pd.DataFrame(final_data)
df.to_csv('GM01.csv', index=False, header=False)



