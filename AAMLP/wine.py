import pandas as pd
df = pd.read_csv("../dataset/archive/winequality-red.csv")

# 一个映射字典，用于将质量值从 0 到 5 进行映射
quality_mapping = {
 3: 0,
 4: 1,
 5: 2,
 6: 3,
 7: 4,
 8: 5
 }
# 你可以使用 pandas 的 map 函数以及任何字典，
# 来转换给定列中的值为字典中的值
df.loc[:, "quality"] = df.quality.map(quality_mapping)
# 使用 frac=1 的 sample 方法来打乱 dataframe
# 由于打乱后索引会改变，所以我们重置索引
df = df.sample(frac=1).reset_index(drop=True)
# 选取前 1000 行作为训练数据
df_train = df.head(1000)
# 选取最后的 599 行作为测试/验证数据
df_test = df.tail(599)


