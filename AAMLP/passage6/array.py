import numpy as np
from sklearn import preprocessing
 # 生成符合均匀分布的随机整数，维度为[1000000, 10000000]
example = np.random.randint(1000, size=1000000)
# 独热编码，非稀疏矩阵
ohe = preprocessing.OneHotEncoder(sparse_output=False)
 # 将随机数组展平
ohe_example = ohe.fit_transform(example.reshape(-1, 1))
# dense_array = ohe_example.toarray()
# print(f"Size of dense array: {dense_array.nbytes} bytes")

print(f"Size of dense array: {ohe_example.nbytes}")
# 独热编码，稀疏矩阵
ohe = preprocessing.OneHotEncoder(sparse_output=True)
 # 将随机数组展平
ohe_example = ohe.fit_transform(example.reshape(-1, 1))
print(f"Size of sparse array: {ohe_example.data.nbytes}")
full_size = (
 ohe_example.data.nbytes +
 ohe_example.indptr.nbytes +
ohe_example.indices.nbytes
)
print(f"Full size of sparse array: {full_size}")